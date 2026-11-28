"""Look to talk: the webcam tells whether you are facing the screen, so speech while you
face it is for Omi and speech while you face away (someone else in the room) is not.
On top of that the lips must be moving, so a video talking on the screen you are watching
does not count: Google's 468-point face mesh (1.2 MB, run by OpenCV, ~4 ms a frame)
measures how far apart your lips are, and talking makes that gap keep changing.

The camera is only on while someone talks: the listener calls wake() when the mic hears
a voice, the camera opens (about 0.3 s to the first reading) and closes again after
HOLD_S of quiet. While it runs, a thread reads a few frames a second, finds the closest
face with OpenCV's YuNet detector (~230 KB, a few ms per frame on CPU), and records
whether its head points at the camera. Frames stay in memory and are dropped right
after; nothing is saved. A closed camera is free for video calls, and a call that holds
it simply means no look triggers, which is right while you talk to people on a call.
"""

from __future__ import annotations

import collections
import contextlib
import fcntl
import hashlib
import logging
import os
import re
import shutil
import threading
import urllib.request
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any

from omi.config import MODELS_DIR

# OpenCV warns on every failed open of a busy camera; the error is reported our way instead.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

log = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "yunet" / "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MESH_PATH = MODELS_DIR / "face-mesh" / "face_landmark.tflite"
MESH_URL = "https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite"
MESH_SHA256 = "1055cb9d4a9ca8b8c688902a3a5194311138ba256bcc94e336d8373a5f30c814"

# The camera only runs while someone talks, so this can be sharp enough for lips.
FRAME_W, FRAME_H = 640, 480
FPS = 10.0  # frames analysed per second; lips move fast
MIN_SCORE = 0.7  # face detector confidence
MIN_FACE_FRAC = 0.08  # face narrower than this share of the frame is someone across the room
# Head turn: nose offset from the eye midpoint, in eye widths. Facing the screen measured
# 0.0-0.25 on a monitor-top webcam; a head turned toward someone beside you goes past 0.4.
MAX_YAW = 0.35
WINDOW_S = 1.0  # how far back "facing the screen right now" looks
MIN_FACING = 0.5  # share of recent frames that must show you facing the screen
HISTORY_S = 120.0
RETRY_S = 1.0  # retry opening a missing or busy camera this often while someone talks
HOLD_S = 15.0  # keep the camera on this long after the last voice, for quick follow-ups
CALL_POLL_S = 2.0  # how often to check whether a call app has the mic
# Lips: openness is the inner-lip gap over the mouth width. Talking keeps it changing; a
# still face, open or closed, keeps it flat. Spread = 90th minus 10th percentile in the
# last LIP_WINDOW_S. Tune with `omi look-test`.
LIP_WINDOW_S = 1.5
MIN_LIP_FRAMES = 4
LIP_MOVE_SPREAD = 0.03  # his talking: mostly 0.02-0.11; still face: 0.005-0.013
MESH_SIZE = 192
_LIP_TOP, _LIP_BOTTOM, _MOUTH_L, _MOUTH_R = 13, 14, 61, 291  # face mesh points

SYS_VIDEO = Path("/sys/class/video4linux")
DEV_BY_ID = Path("/dev/v4l/by-id")
_VIDIOC_QUERYCAP = 0x80685600  # _IOR('V', 0, struct v4l2_capability), 104 bytes
_CAP_VIDEO_CAPTURE = 0x1
_CAP_DEVICE_CAPS = 0x80000000


def _fetch(path: Path, url: str, sha256: str, what: str) -> Path:
    """Download a model on first use and check it is the expected file."""
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    if hashlib.sha256(tmp.read_bytes()).hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{what} download did not match its checksum")
    tmp.replace(path)
    return path


def ensure_model() -> Path:
    """The face detector, and the face mesh the lip check needs."""
    _fetch(MESH_PATH, MESH_URL, MESH_SHA256, "face mesh")
    return _fetch(MODEL_PATH, MODEL_URL, MODEL_SHA256, "face detector")


def closest_face(faces: Any, frame_w: int) -> Any:
    """The nearest clearly seen face, or None. `faces` is YuNet output: rows of
    [x, y, w, h, right eye xy, left eye xy, nose xy, mouth corners xy, score]."""
    if faces is None or not len(faces):
        return None
    face = max(faces, key=lambda f: float(f[2]) * float(f[3]))
    if float(face[14]) < MIN_SCORE or float(face[2]) < MIN_FACE_FRAC * frame_w:
        return None
    return face


def face_yaw(faces: Any, frame_w: int) -> float | None:
    """Head turn of the closest face: nose offset from the eye midpoint, in eye widths
    (0 = straight at the camera). None when no face is clearly seen."""
    face = closest_face(faces, frame_w)
    if face is None:
        return None
    eye_r_x, eye_l_x, nose_x = float(face[4]), float(face[6]), float(face[8])
    eye_dist = abs(eye_l_x - eye_r_x)
    if eye_dist < 1.0:
        return None
    return (nose_x - (eye_r_x + eye_l_x) / 2) / eye_dist


def facing_from_faces(faces: Any, frame_w: int, center: float = 0.0, tolerance: float = MAX_YAW) -> bool:
    """True when the closest face points at the screen. `center` is the head turn that
    means "the screen": 0 when the webcam sits on the monitor you look at, more when it
    sits on another one (set by `omi look-tune`)."""
    yaw = face_yaw(faces, frame_w)
    return yaw is not None and abs(yaw - center) <= tolerance


def lip_openness(landmarks: Any) -> float | None:
    """Inner-lip gap over mouth width from 468 face mesh points (x, y, z rows)."""
    import numpy as np

    pts = np.asarray(landmarks, dtype=np.float32).reshape(-1, 3)[:, :2]
    width = float(np.linalg.norm(pts[_MOUTH_L] - pts[_MOUTH_R]))
    if width < 1.0:
        return None
    return float(np.linalg.norm(pts[_LIP_TOP] - pts[_LIP_BOTTOM])) / width


def lip_spread(values: list[float]) -> float | None:
    """How much the lips moved: 90th minus 10th percentile of openness, or None when too
    few frames saw the mouth (hidden, turned away, camera just started)."""
    if len(values) < MIN_LIP_FRAMES:
        return None
    vals = sorted(values)
    lo = vals[int(0.1 * (len(vals) - 1))]
    hi = vals[round(0.9 * (len(vals) - 1))]
    return hi - lo


def _measure_lips(cv2: Any, mesh: Any, frame: Any, face: Any) -> float | None:
    """Crop the face upright (eyes level, 1.5x the box, as the mesh was trained) and
    read the lip gap. None when the mesh does not see a face there."""
    import math

    import numpy as np

    x, y, w, h = (float(v) for v in face[:4])
    side = max(w, h) * 1.5
    angle = math.degrees(math.atan2(float(face[7]) - float(face[5]), float(face[6]) - float(face[4])))
    cx, cy = x + w / 2, y + h / 2
    m = cv2.getRotationMatrix2D((cx, cy), angle, MESH_SIZE / side)
    m[0, 2] += MESH_SIZE / 2 - cx
    m[1, 2] += MESH_SIZE / 2 - cy
    crop = cv2.warpAffine(frame, m, (MESH_SIZE, MESH_SIZE), borderMode=cv2.BORDER_CONSTANT)
    blob = cv2.dnn.blobFromImage(crop, 1 / 255.0, (MESH_SIZE, MESH_SIZE), swapRB=True)
    mesh.setInput(blob)
    landmarks, flag = mesh.forward(mesh.getUnconnectedOutLayersNames())
    if 1 / (1 + np.exp(-float(np.asarray(flag).reshape(-1)[0]))) < 0.5:
        return None
    return lip_openness(landmarks)


def _captures_video(dev: Path) -> bool | None:
    """Ask the driver whether this node streams pictures (webcams also expose a metadata
    node). Works while another app is using the camera. None when it cannot be asked."""
    try:
        fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        buf = bytearray(104)
        fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf)
    except OSError:
        return None
    finally:
        os.close(fd)
    caps = int.from_bytes(buf[84:88], "little")
    if caps & _CAP_DEVICE_CAPS:
        caps = int.from_bytes(buf[88:92], "little")
    return bool(caps & _CAP_VIDEO_CAPTURE)


def _stable_paths() -> dict[str, str]:
    """/dev/videoN -> its /dev/v4l/by-id link, which survives replugging in another order."""
    out: dict[str, str] = {}
    try:
        links = sorted(DEV_BY_ID.iterdir())
    except OSError:
        return out
    for link in links:
        with contextlib.suppress(OSError):
            out.setdefault(str(link.resolve()), str(link))
    return out


def list_cameras() -> list[dict[str, str]]:
    """Webcams that can stream pictures: [{"id": stable device path, "name": label}]."""
    stable = _stable_paths()
    found: list[tuple[int, str, str]] = []
    try:
        nodes = list(SYS_VIDEO.iterdir())
    except OSError:
        return []
    for node in nodes:
        m = re.fullmatch(r"video(\d+)", node.name)
        if not m:
            continue
        dev = Path("/dev") / node.name
        ok = _captures_video(dev)
        if ok is None:  # cannot ask the driver: fall back to the first node of each device
            with contextlib.suppress(OSError):
                ok = (node / "index").read_text().strip() == "0"
        if not ok:
            continue
        name = ""
        with contextlib.suppress(OSError):
            name = (node / "name").read_text().strip()
        # UVC names are "Product : Maker", cut to 31 characters; the product is enough.
        name = name.split(" : ")[0].strip() or node.name
        found.append((int(m.group(1)), stable.get(str(dev), str(dev)), name))
    found.sort()
    seen: dict[str, int] = {}
    cameras = []
    for _, cam_id, name in found:
        seen[name] = seen.get(name, 0) + 1
        label = name if seen[name] == 1 else f"{name} ({seen[name]})"
        cameras.append({"id": cam_id, "name": label})
    return cameras


def _camera_target(camera: str) -> int | str:
    """What OpenCV opens: the chosen camera, or the first webcam when none is chosen."""
    cam = camera.strip()
    if cam.isdigit():
        return int(cam)
    if cam:
        return cam
    cameras = list_cameras()
    return cameras[0]["id"] if cameras else 0


class LookTracker:
    """Keeps a short history of whether you face the screen, read from the webcam."""

    def __init__(self, camera: str = "", *, clock: Callable[[], float] = monotonic,
                 mic_users: Callable[[], list[str]] | None = None, lip_check: bool = True,
                 lip_threshold: float = LIP_MOVE_SPREAD, yaw_center: float = 0.0,
                 yaw_range: float = MAX_YAW, measure_all: bool = False) -> None:
        self.camera = camera
        self.lip_check = lip_check  # also require moving lips, not just a face toward the screen
        self.lip_threshold = lip_threshold
        self.yaw_center = yaw_center
        self.yaw_range = yaw_range
        self.measure_all = measure_all  # tuning: read lips and head turn of any face seen
        self._clock = clock
        if mic_users is None:
            from omi.audio.pipewire import mic_users
        self._mic_users = mic_users
        self._polled_at = -CALL_POLL_S
        # A call or dictation app is using the mic: look to talk rests and the camera
        # is released, so the call can have it. Holds that app's name.
        self.paused_by = ""
        self._samples: collections.deque[tuple[float, bool, float | None]] = collections.deque(
            maxlen=int(HISTORY_S * FPS))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._woken = threading.Event()
        self._awake_until = 0.0
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.active:
            return
        ensure_model()
        import cv2  # noqa: F401  (fail here, not in the thread, when OpenCV is missing)

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="omi-look", daemon=True)
        self._thread.start()

    def wake(self) -> None:
        """The mic heard a voice: have the camera on, or keep it on, for HOLD_S."""
        self._awake_until = self._clock() + HOLD_S
        self._woken.set()

    @property
    def watching(self) -> bool:
        return not self.paused_by and self._clock() < self._awake_until

    def poll_mic(self, force: bool = False) -> None:
        """Pause while another app records the mic; resume when it stops."""
        now = self._clock()
        if not force and now - self._polled_at < CALL_POLL_S:
            return
        self._polled_at = now
        try:
            users = self._mic_users()
        except Exception:
            users = []
        by = users[0] if users else ""
        if by != self.paused_by:
            if by:
                log.info("look to talk paused: %s is using the microphone", by)
            else:
                log.info("look to talk resumed: the microphone is free")
            self.paused_by = by

    def stop(self) -> None:
        self._stop.set()
        self._woken.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            self._samples.clear()

    def record(self, facing: bool, at: float | None = None, lips: float | None = None,
               yaw: float | None = None) -> None:
        with self._lock:
            self._samples.append((self._clock() if at is None else at, facing, lips, yaw))

    def readings(self, since: float, until: float | None = None) -> list[tuple[bool, float | None, float | None]]:
        """(facing, lip openness, head turn) for each frame in the window."""
        end = self._clock() if until is None else until
        with self._lock:
            return [(f, lip, yaw) for t, f, lip, yaw in self._samples if since <= t <= end]

    def facing_ratio(self, since: float, until: float | None = None) -> float | None:
        """Share of frames between `since` and `until` that show you facing the screen,
        or None when the camera saw nothing in that time (off, busy, unplugged)."""
        end = self._clock() if until is None else until
        with self._lock:
            hits = [f for t, f, _, _ in self._samples if since <= t <= end]
        if not hits:
            return None
        return sum(hits) / len(hits)

    def looking(self, now: float | None = None) -> bool:
        """True when you have been facing the screen for most of the last second."""
        if self.paused_by:
            return False
        end = self._clock() if now is None else now
        ratio = self.facing_ratio(end - WINDOW_S, end)
        return ratio is not None and ratio >= MIN_FACING

    def lips_moving(self, now: float | None = None) -> float | None:
        """Lip spread over the last LIP_WINDOW_S, or None when the mouth was not seen."""
        end = self._clock() if now is None else now
        with self._lock:
            vals = [lip for t, _, lip, _ in self._samples if end - LIP_WINDOW_S <= t <= end and lip is not None]
        return lip_spread(vals)

    def addressed(self, now: float | None = None) -> bool:
        """You are talking to the screen: facing it and, with the lip check on, your lips
        are moving. A hidden mouth means no: say "Hey Omi" instead."""
        if not self.looking(now):
            return False
        if not self.lip_check:
            return True
        spread = self.lips_moving(now)
        return spread is not None and spread >= self.lip_threshold

    def _loop(self) -> None:
        import cv2

        cv2.setNumThreads(1)  # its default thread pool spins idle cores (~45% CPU vs ~5%)
        while not self._stop.is_set():
            self.poll_mic()
            if not self.watching:  # quiet or a call: camera off until the mic hears a voice
                self.error = None  # a busy camera only matters while it is wanted
                self._woken.clear()
                if not self.watching:
                    self._woken.wait(timeout=1.0)
                continue
            target = _camera_target(self.camera)  # re-read each try: a camera may be plugged in later
            cap = cv2.VideoCapture(target, cv2.CAP_V4L2)
            try:
                if not cap.isOpened():
                    raise RuntimeError(f"cannot open camera {target} (missing or in use)")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                detector = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (FRAME_W, FRAME_H), MIN_SCORE, 0.3, 5)
                mesh = cv2.dnn.readNetFromTFLite(str(MESH_PATH)) if self.lip_check else None
                size: tuple[int, int] | None = None
                last = 0.0
                while not self._stop.is_set() and self.watching:
                    self.poll_mic()
                    if not self.watching:
                        break
                    if not cap.grab():  # grab without decoding keeps the newest frame fresh
                        raise RuntimeError("camera stopped sending frames")
                    now = self._clock()
                    if now - last < 1.0 / FPS:
                        continue
                    last = now
                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        continue
                    h, w = frame.shape[:2]
                    if size != (w, h):
                        size = (w, h)
                        detector.setInputSize(size)
                    _, faces = detector.detect(frame)
                    yaw = face_yaw(faces, w)
                    facing = yaw is not None and abs(yaw - self.yaw_center) <= self.yaw_range
                    lips = None
                    if mesh is not None and (facing or (self.measure_all and yaw is not None)):
                        lips = _measure_lips(cv2, mesh, frame, closest_face(faces, w))  # only faces toward us
                    del frame
                    self.record(facing, now, lips, yaw)
                    self.error = None
            except Exception as exc:
                if self.error != str(exc):
                    log.warning("look to talk: %s", exc)
                self.error = str(exc)
            else:
                continue  # quiet again: closed cleanly, nothing to wait out
            finally:
                cap.release()
            self._stop.wait(RETRY_S)
