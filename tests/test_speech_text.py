from hark.speech import ClauseChunker, clean_for_speech


def stream(text: str, step: int = 3, **kw) -> list[str]:
    c = ClauseChunker(**kw)
    out: list[str] = []
    for i in range(0, len(text), step):
        out += c.feed(text[i : i + step])
    return out + c.flush()


def test_first_chunk_at_first_clause_then_sentences():
    assert stream("Sure, I can do that. It is done, and more, too. Bye") == [
        "Sure,",
        "I can do that.",
        "It is done, and more, too.",
        "Bye",
    ]


def test_first_chunk_emitted_before_rest_arrives():
    c = ClauseChunker()
    assert c.feed("Okay, so") == ["Okay,"]
    assert c.feed(" the plan is") == []


def test_first_chunk_after_eight_words_without_punctuation():
    chunks = stream("one two three four five six seven eight nine ten. eleven")
    assert chunks[0] == "one two three four five six seven eight"
    assert chunks[1] == "nine ten."


def test_decimals_abbreviations_initials_times():
    text = "Dr. Smith said pi is 3.14 e.g. in class at 3:30 with J. R. Tolkien. Next one."
    assert stream(text, step=1) == [
        "Dr. Smith said pi is 3.14 e.g. in",
        "class at 3:30 with J. R. Tolkien.",
        "Next one.",
    ]


def test_ellipsis_is_clause_not_sentence():
    assert stream("Hello. Well... maybe later. Done") == ["Hello.", "Well... maybe later.", "Done"]


def test_question_and_exclamation_and_quotes():
    assert stream('Hi! He said "stop." Then? ok') == ["Hi!", 'He said "stop."', "Then?", "ok"]


def test_waits_on_trailing_punctuation():
    c = ClauseChunker()
    assert c.feed("Version 3.") == []  # could be 3.5
    assert c.feed("5 is out") == []
    assert c.flush() == ["Version 3.5 is out"]


def test_first_sentence_end_after_a_word_cuts_at_once():
    c = ClauseChunker()
    assert c.feed("Hello!") == ["Hello!"]  # no wait for the next token
    assert c.feed(" Dr.") == []  # abbreviation: still waits
    assert c.feed(" Who is out.") == []  # later sentences still wait
    assert c.flush() == ["Dr. Who is out."]


def test_code_fences_dropped_even_when_split():
    text = "Here: ``" + "`python\nprint('x. y. z.')\n``` that was it. Bye."
    assert stream(text, step=2) == ["Here:", "that was it.", "Bye."]


def test_long_sentence_breaks_at_clause_after_max_words():
    words = " ".join(f"w{i}" for i in range(12))
    chunks = stream(f"Start. {words}, {words}, end.", max_words=10)
    assert chunks[0] == "Start."
    assert len(chunks) >= 3


def test_flush_on_empty():
    c = ClauseChunker()
    assert c.flush() == []


def test_clean_markdown_and_links():
    md = "## Title\n- **bold** item with `code` and [the docs](http://x.com)\n> quoted"
    assert clean_for_speech(md) == "Title. bold item with code and the docs. quoted"


def test_clean_urls_and_code_blocks():
    assert clean_for_speech("See https://example.com/a?b=1. Done") == "See a link. Done"
    assert clean_for_speech("Run ```bash\nrm -rf /\n``` now") == "Run now"
    assert clean_for_speech("www.example.org") == "a link"


def test_clean_emoji_and_symbols():
    text = "Great \U0001f389 it costs $5 & 50% -> ok — range 5-10 at 20°C, ~3 items #4"
    assert clean_for_speech(text) == (
        "Great it costs 5 dollars and 50 percent to ok, range 5 to 10 at 20 degrees Celsius, about 3 items number 4"
    )


def test_clean_keeps_plain_text_and_identifiers():
    assert clean_for_speech("Plain text, nothing to do.") == "Plain text, nothing to do."
    assert clean_for_speech("open file_name now") == "open file name now"
    assert clean_for_speech("mail me@example.com") == "mail me at example.com"
