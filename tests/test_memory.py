"""Memory tests. Everything lives in tmp_path; the real agent-memory folder is never touched."""

from __future__ import annotations

import asyncio
import builtins
import io
import json
import os
import pathlib
import statistics
import threading
import time

import pytest

from omi.config import MemoryConfig
from omi.memory import FileMemory, NullMemory, create
from omi.memory.index import build_query
from omi.memory.jobs import (
    apply_ops,
    digest,
    end_session,
    extract_facts,
    parse_ops,
    rebuild_profile,
)
from omi.memory.store import parse_frontmatter, slugify
from omi.memory.tools import memory_tools


def write_shared(folder: pathlib.Path, fname: str, name: str, desc: str, body: str, type_="user"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / fname).write_text(
        f"---\nname: {name}\ndescription: {desc}\nmetadata:\n  type: {type_}\n---\n\n{body}\n"
    )


@pytest.fixture
def cfg(tmp_path):
    return MemoryConfig(
        dir=str(tmp_path / "shared" / "voice"),
        shared_dir=str(tmp_path / "shared"),
        index=str(tmp_path / "data" / "memory.sqlite"),
    )


@pytest.fixture
def mem(cfg):
    return create(cfg)


def snapshot(folder: pathlib.Path) -> dict:
    return {
        str(p.relative_to(folder)): (p.stat().st_mtime_ns, p.read_bytes())
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


# --- facts: round trip, collisions, odd characters ---


def test_save_recall_forget_round_trip(mem, cfg):
    path = mem.save_fact("Coffee order", "The user drinks oat milk flat whites.", "coffee", "user")
    assert pathlib.Path(path).parent == pathlib.Path(cfg.dir) / "facts"
    hits = mem.recall("what coffee do I drink")
    assert [h.name for h in hits] == ["Coffee order"]
    assert hits[0].kind == "fact" and "oat milk" in hits[0].text

    assert mem.forget("coffee") == ["Coffee order"]
    assert mem.recall("coffee") == []
    assert not pathlib.Path(path).exists()
    trashed = list((pathlib.Path(cfg.dir) / ".trash").iterdir())
    assert len(trashed) == 1 and "oat milk" in trashed[0].read_text()


def test_fact_file_format(mem):
    path = pathlib.Path(mem.save_fact("Editor: Neovim", "Uses Neovim: always.", "tools", "feedback"))
    meta, body = parse_frontmatter(path.read_text())
    assert meta["name"] == "Editor: Neovim"
    assert meta["metadata"]["type"] == "feedback"
    assert len(meta["updated"]) == 10 and meta["updated"][4] == "-"
    assert body.strip() == "Uses Neovim: always."
    assert not [p for p in path.parent.iterdir() if p.name.startswith(".tmp")]


def test_same_name_updates_in_place(mem):
    a = mem.save_fact("Pet", "A dog named Rex.")
    b = mem.save_fact("pet", "A dog named Max.")
    assert a == b
    assert len(list(pathlib.Path(a).parent.glob("*.md"))) == 1
    assert "Max" in mem.recall("dog")[0].text


def test_slug_collisions_and_odd_characters(mem):
    p1 = mem.save_fact("C++ fan", "likes c plus plus")
    p2 = mem.save_fact("C# fan", "likes c sharp")
    p3 = mem.save_fact("c fan", "likes plain c")
    assert len({p1, p2, p3}) == 3
    assert {pathlib.Path(p).name for p in (p1, p2, p3)} == {"c-fan.md", "c-fan-2.md", "c-fan-3.md"}
    nasty = mem.save_fact("../../etc/passwd éè \U0001f600 \n tab\t", "body")
    assert pathlib.Path(nasty).parent == pathlib.Path(p1).parent
    assert slugify("\U0001f600\U0001f600") == "fact"
    assert len(slugify("x" * 500)) <= 60
    names = {h.name for h in mem.recall("sharp plain")}
    assert {"C# fan", "c fan"} <= names


# --- the index ---


def test_index_rebuilds_after_sqlite_deleted(cfg):
    m1 = create(cfg)
    m1.save_fact("Bike", "Rides a red gravel bike on weekends.")
    m1._index.close()
    for suffix in ("", "-wal", "-shm"):
        pathlib.Path(cfg.index + suffix).unlink(missing_ok=True)
    m2 = create(cfg)
    assert [h.name for h in m2.recall("gravel bike")] == ["Bike"]


def test_corrupt_sqlite_is_rebuilt(cfg):
    m1 = create(cfg)
    m1.save_fact("Bike", "Rides a red gravel bike.")
    m1._index.close()
    pathlib.Path(cfg.index).write_bytes(b"this is not a database" * 100)
    for suffix in ("-wal", "-shm"):
        pathlib.Path(cfg.index + suffix).unlink(missing_ok=True)
    assert [h.name for h in create(cfg).recall("gravel")] == ["Bike"]


def test_index_picks_up_hand_edit(cfg):
    m = create(cfg)
    path = pathlib.Path(m.save_fact("Plant", "Has a ficus."))
    assert m.recall("monstera") == []
    text = path.read_text().replace("Has a ficus.", "Has a monstera.")
    path.write_text(text)
    os.utime(path, ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))
    assert m.sync() == 1  # the running process notices on the next sync
    assert [h.name for h in m.recall("monstera")] == ["Plant"]
    assert m.recall("ficus") == []
    # and a restart notices too
    path.write_text(text.replace("monstera", "cactus"))
    os.utime(path, ns=(time.time_ns() + 2 * 10**9, time.time_ns() + 2 * 10**9))
    assert [h.name for h in create(cfg).recall("cactus")] == ["Plant"]


def test_hand_deleted_and_hand_created_files(cfg):
    m = create(cfg)
    gone = pathlib.Path(m.save_fact("Old", "old stuff about zebras"))
    gone.unlink()
    (gone.parent / "new.md").write_text("---\nname: Fresh\ndescription: x\n---\n\nlikes giraffes\n")
    m.sync()
    assert m.recall("zebras") == []
    assert [h.name for h in m.recall("giraffes")] == ["Fresh"]


def test_recall_edge_queries(mem):
    mem.save_fact("Tea", "Likes green tea in the morning.")
    assert mem.recall("") == []
    assert mem.recall("the and of to a is") == []
    assert mem.recall("   !!! ??? ") == []
    for nasty in ['tea" OR *', "NOT tea AND (", "tea NEAR/3 x", "col:tea", '"', "-tea", "tea*"]:
        mem.recall(nasty)  # must never raise
    assert mem.recall("TEA")[0].name == "Tea"
    assert mem.recall("teas")[0].name == "Tea"
    assert build_query("What's the weather, please?") == '"weather"*'
    assert build_query("a the of") == ""
    assert build_query("Gre en") == '"gre"* OR "en"*'


# --- the shared folder ---


def test_shared_searched_never_written_and_private_skipped(tmp_path, cfg):
    shared = pathlib.Path(cfg.shared_dir)
    write_shared(shared, "wifi.md", "Home setup", "network layout", "Router is a mesh of three nodes.")
    write_shared(shared, "private-notes.md", "Notes", "stuff", "mesh secret alpaca")
    write_shared(shared, "x.md", "Personal-Details card", "d", "mesh alpaca street")
    write_shared(shared, "y.md", "Contact", "has personal-details inside", "mesh alpaca phone")
    write_shared(shared, "z.md", "Money", "Private accounts", "mesh alpaca bank")
    write_shared(shared, "MEMORY.md", "Index", "index", "mesh alpaca index")
    write_shared(shared / "voice", "own.md", "Voice thing", "d", "mesh alpaca voice")
    write_shared(shared / "otherdir", "sub.md", "Sub thing", "d", "mesh alpaca sub")
    (shared / "nofm.md").write_text("mesh alpaca no frontmatter at all")
    before = snapshot(shared)

    m = create(cfg)
    hits = m.recall("mesh")
    assert [h.name for h in hits] == ["Home setup"]
    assert m.recall("alpaca") == []

    # writing own facts and forgetting never touches shared files
    m.save_fact("Mine", "my own mesh fact")
    assert m.forget("router nodes") == []  # only a shared fact matches: not removable
    m.log_turn("s1", "user", "hello mesh")
    m.end_session("s1", "- talked")
    m.sync()
    after = snapshot(shared)
    own = {k for k in after if k.startswith("voice" + os.sep)}
    assert {k: v for k, v in after.items() if k not in own and k not in before} == {}
    assert {k: after[k] for k in before if not k.startswith("voice" + os.sep)} == {
        k: v for k, v in before.items() if not k.startswith("voice" + os.sep)
    }
    assert [h.name for h in m.recall("router")] == ["Home setup"]


def test_private_file_edited_to_public_is_picked_up(cfg):
    shared = pathlib.Path(cfg.shared_dir)
    write_shared(shared, "a.md", "Hobby", "private hobby", "collects stamps")
    m = create(cfg)
    assert m.recall("stamps") == []
    write_shared(shared, "a.md", "Hobby", "a hobby", "collects stamps")
    os.utime(shared / "a.md", ns=(time.time_ns() + 10**9,) * 2)
    m.sync()
    assert [h.name for h in m.recall("stamps")] == ["Hobby"]


# --- sessions ---


def test_session_transcript_digest_and_carryover(mem, cfg):
    mem.log_turn("s1", "user", "Please set a timer for the pasta\nthree lines")
    mem.log_turn("s1", "assistant", "Timer set for ten minutes.")
    sessions = list((pathlib.Path(cfg.dir) / "sessions").glob("*.md"))
    assert len(sessions) == 1 and sessions[0].name.count("-") >= 2
    assert "## Digest" not in sessions[0].read_text()
    assert mem.transcript("s1").splitlines() == [
        "user: Please set a timer for the pasta three lines",
        "assistant: Timer set for ten minutes.",
    ]
    mem.end_session("s1", "- Set a ten minute pasta timer.")
    text = sessions[0].read_text()
    assert text.index("user: Please") < text.index("## Digest") < text.index("pasta timer")
    assert [h.kind for h in mem.recall("pasta timer")] == ["session"]
    assert "pasta timer" in mem.context()
    assert "pasta timer" in (pathlib.Path(cfg.dir) / "carryover.md").read_text()


def test_carryover_keeps_last_five(mem, cfg):
    for i in range(8):  # all in the same second: headings must still be distinct
        mem.log_turn(f"s{i}", "user", f"hi {i}")
        mem.end_session(f"s{i}", f"- d{i}")
    carry = (pathlib.Path(cfg.dir) / "carryover.md").read_text()
    assert carry.count("### ") == 5
    assert all(f"- d{i}\n" in carry for i in range(3, 8))
    assert all(f"- d{i}\n" not in carry for i in range(3))


def test_carryover_token_budget(mem, cfg):
    for i in range(8):
        mem.log_turn(f"s{i}", "user", "hi")
        mem.end_session(f"s{i}", f"- digest number {i} " + "word " * 40)
    carry = (pathlib.Path(cfg.dir) / "carryover.md").read_text()
    assert 1 <= carry.count("### ") <= 5
    assert "digest number 7" in carry and "digest number 0" not in carry
    assert len(carry) <= 300 * 4 + 1


def test_late_turn_after_end_session_is_kept(mem, cfg):
    mem.log_turn("s", "user", "first line")
    mem.end_session("s", "- one")
    mem.log_turn("s", "user", "late line")
    mem.end_session("s", "- two")
    files = list((pathlib.Path(cfg.dir) / "sessions").glob("*.md"))
    blob = "\n".join(f.read_text() for f in files)
    assert "first line" in blob and "late line" in blob and len(files) == 2
    assert all(f.read_text().count("## Digest") == 1 for f in files)


def test_new_process_finds_session_by_id(cfg):
    m1 = create(cfg)
    m1.log_turn("abc", "user", "hello there")
    m2 = create(cfg)
    assert "hello there" in m2.transcript("abc")
    m2.end_session("abc", "- greeted")
    assert len(list((pathlib.Path(cfg.dir) / "sessions").glob("*.md"))) == 1


def test_concurrent_log_turn(mem, cfg):
    def worker(sid: str, n: int):
        for i in range(n):
            mem.log_turn(sid, "user", f"{sid} line {i} " + "x" * 200)

    threads = [threading.Thread(target=worker, args=(f"s{t % 3}", 50)) for t in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = 0
    for sid in ("s0", "s1", "s2"):
        lines = mem.transcript(sid).splitlines()
        assert all(ln.startswith("user: s") and ln.endswith("x" * 200) for ln in lines)
        total += len(lines)
    assert total == 12 * 50
    assert len(list((pathlib.Path(cfg.dir) / "sessions").glob("*.md"))) == 3


# --- context and disabled ---


def test_context_does_no_io(mem, monkeypatch):
    mem.write_profile("Likes short answers.")
    mem.log_turn("s", "user", "x")
    mem.end_session("s", "- did a thing")

    def boom(*a, **k):
        raise AssertionError("I/O in context()")

    monkeypatch.setattr(builtins, "open", boom)
    monkeypatch.setattr(io, "open", boom)
    monkeypatch.setattr(os, "open", boom)
    monkeypatch.setattr(os, "stat", boom)
    monkeypatch.setattr(pathlib.Path, "read_text", boom)
    monkeypatch.setattr(pathlib.Path, "open", boom)
    first = mem.context()
    assert "Likes short answers." in first and "did a thing" in first
    assert mem.context() == first


def test_context_refreshes_only_after_writes(mem):
    assert mem.context() == ""
    mem.save_fact("A", "b")
    assert mem.context() == ""
    mem.write_profile("Prefers metric units.")
    assert "metric" in mem.context()


def test_disabled_is_noop(cfg):
    cfg.enabled = False
    m = create(cfg)
    assert isinstance(m, NullMemory)
    assert m.context() == "" and m.recall("x") == [] and m.forget("x") == []
    m.log_turn("s", "user", "x")
    m.end_session("s", "d")
    assert not pathlib.Path(cfg.dir).exists() and not pathlib.Path(cfg.index).exists()


# --- timing ---


def test_recall_timing_2000_facts(cfg, capsys):
    words = ("alpha beta gamma delta epsilon zeta theta kappa lambda sigma omega piano guitar "
             "coffee garden bicycle recipe holiday project server kernel window music novel").split()
    shared = pathlib.Path(cfg.shared_dir)
    for i in range(2000):
        w = [words[(i * 7 + j * 3) % len(words)] for j in range(12)]
        write_shared(shared, f"f{i}.md", f"Fact {i} {w[0]}", f"about {w[1]} {w[2]}", " ".join(w) + f" item{i}")
    t0 = time.perf_counter()
    m = create(cfg)
    build_ms = (time.perf_counter() - t0) * 1000
    assert len(m.facts()) == 2000
    queries = ["piano lessons", "what coffee do I like", "garden bicycle", "project server kernel",
               "item1234", "holiday recipe novel music", "omega", "nothing matches xyzzy"]
    for q in queries * 3:
        m.recall(q)  # warm
    times = []
    for _ in range(40):
        for q in queries:
            t = time.perf_counter()
            m.recall(q)
            times.append((time.perf_counter() - t) * 1000)
    times.sort()
    p50, p95 = statistics.median(times), times[int(len(times) * 0.95)]
    t0 = time.perf_counter()
    create(cfg)  # second start: nothing changed, only stats
    restart_ms = (time.perf_counter() - t0) * 1000
    with capsys.disabled():
        print(f"\nrecall 2000 facts: p50={p50:.3f} ms p95={p95:.3f} ms max={times[-1]:.3f} ms "
              f"| first build {build_ms:.0f} ms, warm restart {restart_ms:.0f} ms")
    assert p95 < 10


# --- tools ---


def run(coro):
    return asyncio.run(coro)


def test_memory_tools(mem):
    tools = {t.name: t for t in memory_tools(mem)}
    assert set(tools) == {"remember", "recall", "forget"}
    assert tools["remember"].safety == "reversible"
    assert tools["recall"].safety == "read"
    assert tools["forget"].safety == "reversible"
    for t in tools.values():
        assert t.parameters["type"] == "object" and t.parameters["required"]

    r = run(tools["remember"].fn(fact="I take my tea with honey and no milk"))
    assert r.ok and r.text == "Got it."
    r = run(tools["recall"].fn(query="tea honey"))
    assert r.ok and "honey" in r.text
    r = run(tools["recall"].fn(query="submarines"))
    assert r.ok and "don't have" in r.text
    r = run(tools["forget"].fn(query="tea honey"))
    assert r.ok and r.text.startswith("Forgot ")
    assert run(tools["recall"].fn(query="tea honey")).text.startswith("I don't have")
    assert "couldn't find" in run(tools["forget"].fn(query="tea honey")).text
    assert not run(tools["remember"].fn(fact="   ")).ok


# --- jobs ---


def fake(*answers: str):
    """A fake LLM that returns the answers in order and records the prompts."""
    prompts: list[str] = []
    queue = list(answers)

    async def llm(prompt: str) -> str:
        prompts.append(prompt)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    llm.prompts = prompts  # type: ignore[attr-defined]
    return llm


def test_digest_bullets():
    llm = fake("Here you go:\n1. asked for a timer\n* got a ten minute timer\n- wants pasta\n- four\n- five\n- six\n")
    out = run(digest(llm, "user: timer please"))
    assert out.splitlines() == [
        "- asked for a timer", "- got a ten minute timer", "- wants pasta", "- four", "- five"
    ]
    assert run(digest(fake("Talked about timers.\nSet one."), "x")) == "- Talked about timers.\n- Set one."
    assert run(digest(fake(""), "x")) == ""


def test_digest_truncates_long_transcripts():
    llm = fake("- ok")
    transcript = "\n".join(f"user: line {i} " + "z" * 50 for i in range(500))
    run(digest(llm, transcript))
    prompt = llm.prompts[0]
    assert len(prompt) < 7000 and "line 499" in prompt and "line 0 " not in prompt


GOOD = [{"op": "ADD", "name": "Likes jazz", "description": "music taste", "type": "user",
         "body": "The user likes jazz."}]


@pytest.mark.parametrize(
    "answer",
    [
        json.dumps(GOOD),
        "```json\n" + json.dumps(GOOD) + "\n```",
        "Sure! Here are the ops:\n" + json.dumps(GOOD) + "\nHope that helps [1].",
        "<think>maybe [1, 2]</think>" + json.dumps(GOOD),
        json.dumps({"ops": GOOD}),
        json.dumps(GOOD[0]),
        'junk {not json} ' + json.dumps(GOOD) + " trailing }",
        json.dumps(GOOD).replace('"ADD"', '"add"'),
    ],
)
def test_extract_facts_tolerates_wrapping(answer):
    ops = run(extract_facts(fake(answer), "- listened to jazz", ["Pet: a dog"]))
    assert ops == [{"op": "ADD", "name": "Likes jazz", "description": "music taste",
                    "type": "user", "body": "The user likes jazz."}]


@pytest.mark.parametrize(
    "answer",
    ["", "no json at all", "[{", '[{"op": "ADD", "name": "x"', '{"op": "DELETE", "name": "a", "body": "b"}',
     '[{"op": "ADD", "name": "", "body": "b"}]', "[1, 2, 3]", "null", '["ADD"]'],
)
def test_extract_facts_malformed_gives_empty(answer):
    assert run(extract_facts(fake(answer), "d", [])) == []


def test_extract_ops_partial_and_type_fix():
    ops = parse_ops('[{"op":"ADD","name":"A","body":"b","type":"weird"}, 5, {"op":"NOOP"}, {"op":"UPDATE"}]')
    assert [o["op"] for o in ops] == ["ADD", "NOOP"]
    assert ops[0]["type"] == "user"


def test_extraction_prompt_has_the_rules():
    llm = fake("[]")
    run(extract_facts(llm, "- digest text", ["Pet: a dog"]))
    p = llm.prompts[0].lower()
    for needle in ("durable facts", "preferences", "ongoing projects", "never store secrets",
                   "passwords", "keys", "health", "financial", "one-off", "prefer update over add",
                   "pet: a dog", "digest text"):
        assert needle in p


def test_apply_ops(mem):
    ops = parse_ops(json.dumps([
        {"op": "ADD", "name": "Pet", "body": "A dog named Rex.", "type": "user"},
        {"op": "UPDATE", "name": "pet", "body": "A dog named Rex and a cat.", "type": "user"},
        {"op": "NOOP"},
        {"op": "ADD", "name": "Wifi", "body": "The wifi password is hunter2.", "type": "reference"},
    ]))
    assert apply_ops(mem, ops) == ["Pet", "pet"]
    assert len(mem.facts()) == 1 and "cat" in mem.recall("rex")[0].text
    assert mem.recall("hunter2") == []


def test_end_session_job(mem, cfg):
    mem.save_fact("Jazz", "The user likes jazz.", "music")
    mem.log_turn("s1", "user", "Play some jazz and remember I am learning Rust")
    mem.log_turn("s1", "assistant", "Playing jazz.")
    llm = fake(
        "- Asked for jazz\n- Is learning Rust",
        '```json\n[{"op":"UPDATE","name":"Jazz","description":"music","type":"user",'
        '"body":"The user likes jazz and is learning Rust."}]\n```',
    )
    out = run(end_session(mem, "s1", llm))
    assert out == "- Asked for jazz\n- Is learning Rust"
    assert "Jazz: music: The user likes jazz." in llm.prompts[1]  # similar existing fact was passed in
    assert "learning Rust" in (pathlib.Path(cfg.dir) / "carryover.md").read_text()
    assert [h.name for h in mem.recall("rust") if h.kind == "fact"] == ["Jazz"]
    assert len(list((pathlib.Path(cfg.dir) / "facts").glob("*.md"))) == 1


def test_end_session_job_survives_bad_llm(mem):
    mem.log_turn("s1", "user", "hello")
    assert run(end_session(mem, "s1", fake("- did a thing", "total garbage {{{"))) == "- did a thing"
    assert mem.facts() == []

    async def broken(prompt: str) -> str:
        raise RuntimeError("model down")

    mem.log_turn("s2", "user", "hello again")
    assert run(end_session(mem, "s2", broken)) == ""
    assert run(end_session(mem, "never-logged", fake("- x"))) == ""


def test_rebuild_profile(mem, cfg):
    assert run(rebuild_profile(mem, fake("nothing"))) == ""  # no facts yet
    mem.save_fact("Likes brevity", "Wants short answers.", "style", "feedback")
    mem.save_fact("Name", "The user is called Sam.", "identity", "user")
    mem.save_fact("Omi project", "Building a voice assistant.", "project", "project")
    llm = fake("```\n" + "Sam wants short answers. " * 400 + "\n```")
    out = run(rebuild_profile(mem, llm))
    assert 0 < len(out) <= 500 * 4
    assert "Wants short answers." in llm.prompts[0] and "called Sam" in llm.prompts[0]
    assert "Building a" not in llm.prompts[0]  # project facts are not profile material
    assert (pathlib.Path(cfg.dir) / "profile.md").read_text().strip() == out
    assert out in mem.context()


# --- fixes ---


def test_remember_refuses_secrets_and_keeps_numbers_out_of_names(mem, cfg):
    tools = {t.name: t for t in memory_tools(mem)}
    r = run(tools["remember"].fn(fact="The wifi password is hunter2 and my github token is ghp_ABCDEFGHIJKLMNOPQRSTUVWX12"))
    assert not r.ok and "secret" in r.text
    facts = pathlib.Path(cfg.dir) / "facts"
    assert not facts.exists() or not list(facts.iterdir())
    assert mem.save_fact("key", "my api key is sk-abcdefghijklmnopqrstuvwx") == ""
    assert run(tools["remember"].fn(fact="Sam's locker number is 4471 at the gym")).ok
    (name,) = [p.name for p in facts.iterdir()]
    assert not any(ch.isdigit() for ch in name)


def test_logged_turns_are_redacted_and_recallable_at_once(mem, cfg):
    mem.log_turn("s1", "user", "my password is hunter2, remember I see Doctor Okafor on friday")
    text = next((pathlib.Path(cfg.dir) / "sessions").iterdir()).read_text()
    assert "hunter2" not in text and "password is [redacted]" in text
    assert any("Okafor" in h.text for h in mem.recall("dentist Okafor"))


def test_autosync_sees_new_and_rewritten_files(cfg, tmp_path):
    shared = pathlib.Path(cfg.shared_dir)
    write_shared(shared, "tea.md", "Tea", "drink", "Sam drinks green tea every morning.")
    m = FileMemory(cfg)
    m.start_autosync(0.05)
    try:
        write_shared(shared, "bike.md", "Bike", "ride", "Sam rides a red bicycle.")
        write_shared(shared, "tea.md", "Tea", "drink", "Sam now drinks coffee every morning.")
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not (m.recall("bicycle") and m.recall("coffee")):
            time.sleep(0.02)
        assert m.recall("bicycle") and m.recall("coffee")
        assert not any("green tea" in h.text for h in m.recall("tea"))
    finally:
        m.stop_autosync()


def test_frontmatter_block_scalars_and_flow_maps():
    meta, body = parse_frontmatter(
        "---\nname: Coffee\ndescription: >-\n  Takes coffee\n  black.\nnotes: |\n  one\n  two\n"
        "metadata: {type: feedback, source: x}\n---\nBody")
    assert meta["description"] == "Takes coffee black." and meta["notes"] == "one\ntwo"
    assert meta["metadata"] == {"type": "feedback", "source": "x"} and body == "Body"


def test_end_session_rebuilds_the_profile_when_facts_change(mem, cfg):
    mem.log_turn("s1", "user", "I moved to Lisbon.")
    llm = fake("- Sam moved to Lisbon.",
               '[{"op": "ADD", "name": "Home city", "description": "where Sam lives", "type": "user", '
               '"body": "Sam lives in Lisbon."}]',
               "Sam lives in Lisbon.")
    run(end_session(mem, "s1", llm))
    assert len(llm.prompts) == 3 and "Write a short profile" in llm.prompts[2]
    assert "Lisbon" in (pathlib.Path(cfg.dir) / "profile.md").read_text()
    # nothing new: no profile rewrite
    mem.log_turn("s2", "user", "hi")
    llm2 = fake("- Said hi.", '[{"op": "NOOP"}]')
    run(end_session(mem, "s2", llm2))
    assert len(llm2.prompts) == 2


def test_profile_written_at_startup_only_when_missing(mem, cfg):
    from omi.memory.jobs import profile_if_missing

    llm = fake("Sam likes tea.")
    assert run(profile_if_missing(mem, llm)) == "" and llm.prompts == []  # no facts yet
    mem.save_fact("Tea", "Sam likes tea.")
    assert run(profile_if_missing(mem, llm)) == "Sam likes tea."
    assert run(profile_if_missing(mem, llm)) == "" and len(llm.prompts) == 1  # profile exists now
