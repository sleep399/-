"""Thread-safe shared capture for RTSP/HTTP(S)/RTMP camera streams.

One capture worker is created per normalized URL.  Subscribers consume the
latest frame independently, so a slow recognizer cannot hold up capture or
other recognition modules.
"""

from __future__ import annotations

import ipaddress
import threading
import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from app.utils.video import validate_stream_url


class NetworkStreamError(RuntimeError):
    """Raised when a shared network stream cannot be opened or read."""


def normalize_stream_url(url: str) -> str:
    """Validate and canonicalize a supported camera URL for registry lookup."""
    validated = validate_stream_url(url)
    if len(validated) > 2048:
        raise ValueError("network camera URL is too long")
    parsed = urlsplit(validated)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and (
        address.is_link_local or address.is_multicast or address.is_unspecified
    ):
        raise ValueError("network camera address is not allowed")
    if ":" in hostname:  # urlsplit removes IPv6 brackets.
        hostname = f"[{hostname}]"

    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo += f":{parsed.password}"
        userinfo += "@"

    port = parsed.port
    default_port = {"http": 80, "https": 443, "rtsp": 554, "rtmp": 1935}.get(scheme)
    port_suffix = f":{port}" if port is not None and port != default_port else ""
    netloc = f"{userinfo}{hostname}{port_suffix}"
    path = parsed.path or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


@dataclass(frozen=True)
class StreamFrame:
    sequence: int
    frame: np.ndarray
    captured_at: float


@dataclass(frozen=True)
class StreamInfo:
    width: int
    height: int
    fps: float


class _CaptureWorker:
    def __init__(
        self,
        url: str,
        capture_factory: Callable[..., object],
    ) -> None:
        self.url = url
        self._capture_factory = capture_factory
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._capture_lock = threading.Lock()
        self._capture = None
        self._capture_released = False
        self._sequence = 0
        self._latest_frame: np.ndarray | None = None
        self._captured_at = 0.0
        self._error: str | None = None
        self._stream_info: StreamInfo | None = None
        self._subscribers = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"network-camera-{abs(hash(url)) & 0xFFFF:x}",
            daemon=True,
        )
        self._thread.start()

    def add_subscriber(self) -> None:
        with self._condition:
            if self._stop_event.is_set():
                raise NetworkStreamError("stream is closing")
            self._subscribers += 1

    def remove_subscriber(self) -> int:
        with self._condition:
            self._subscribers = max(0, self._subscribers - 1)
            return self._subscribers

    def subscriber_count(self) -> int:
        with self._condition:
            return self._subscribers

    def wait_until_ready(self, timeout: float | None = 8.5) -> StreamInfo:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while (
                self._stream_info is None
                and not self._error
                and not self._stop_event.is_set()
            ):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise NetworkStreamError("network camera stream open timed out")
                self._condition.wait(remaining)

            if self._stream_info is not None:
                return self._stream_info
            if self._error:
                raise NetworkStreamError(self._error)
            raise NetworkStreamError("network camera stream is closing")

    def read_after(self, sequence: int, timeout: float | None = 1.0) -> StreamFrame | None:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._sequence <= sequence and not self._error and not self._stop_event.is_set():
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)

            if self._sequence > sequence and self._latest_frame is not None:
                # Each recognizer owns its frame. This prevents an annotator or
                # model from mutating the shared buffer seen by other modules.
                return StreamFrame(
                    sequence=self._sequence,
                    frame=self._latest_frame.copy(),
                    captured_at=self._captured_at,
                )
            if self._error:
                raise NetworkStreamError(self._error)
            return None

    def stop(self) -> None:
        self._stop_event.set()
        self._release_capture()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    def _release_capture(self) -> None:
        with self._capture_lock:
            if self._capture is not None and not self._capture_released:
                self._capture_released = True
                try:
                    self._capture.release()
                except Exception:
                    pass

    def _set_error(self, message: str) -> None:
        with self._condition:
            self._error = message
            self._condition.notify_all()

    @staticmethod
    def _capture_value(capture, prop: int, default: float) -> float:
        try:
            value = float(capture.get(prop))
        except Exception:
            return default
        if not np.isfinite(value) or value <= 0:
            return default
        return value

    def _set_ready(self, capture) -> None:
        info = StreamInfo(
            width=int(self._capture_value(capture, cv2.CAP_PROP_FRAME_WIDTH, 1280)),
            height=int(self._capture_value(capture, cv2.CAP_PROP_FRAME_HEIGHT, 720)),
            fps=self._capture_value(capture, cv2.CAP_PROP_FPS, 25.0),
        )
        with self._condition:
            if self._stop_event.is_set():
                return
            self._stream_info = info
            self._condition.notify_all()

    def _run(self) -> None:
        try:
            capture = self._capture_factory(self.url, cv2.CAP_FFMPEG)
            with self._capture_lock:
                self._capture = capture
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if not capture.isOpened():
                self._set_error("unable to open network camera stream")
                return
            self._set_ready(capture)

            consecutive_read_failures = 0
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= 3:
                        if not self._stop_event.is_set():
                            self._set_error("network camera stream read failed or ended")
                        return
                    if self._stop_event.wait(0.1):
                        return
                    continue
                consecutive_read_failures = 0
                with self._condition:
                    self._sequence += 1
                    self._latest_frame = frame.copy()
                    self._captured_at = time.time()
                    self._condition.notify_all()
        except Exception as exc:
            if not self._stop_event.is_set():
                # Capture backends sometimes include the full credential-bearing
                # URL in exception text. Keep client/log-facing errors sanitized.
                self._set_error(
                    f"network camera capture failed ({type(exc).__name__})"
                )
        finally:
            self._release_capture()
            with self._condition:
                self._condition.notify_all()


class NetworkStreamSubscription:
    def __init__(self, hub: "NetworkStreamHub", key: str, worker: _CaptureWorker) -> None:
        self._hub = hub
        self._key = key
        self._worker = worker
        self._sequence = 0
        self._closed = False
        self._close_lock = threading.Lock()

    @property
    def normalized_url(self) -> str:
        return self._key

    def next_frame(self, timeout: float | None = 1.0) -> StreamFrame | None:
        if self._closed:
            raise NetworkStreamError("subscription is closed")
        item = self._worker.read_after(self._sequence, timeout)
        if item is not None:
            self._sequence = item.sequence
        return item

    def wait_until_ready(self, timeout: float | None = 8.5) -> StreamInfo:
        if self._closed:
            raise NetworkStreamError("subscription is closed")
        return self._worker.wait_until_ready(timeout)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._hub._unsubscribe(self._key, self._worker)

    def __enter__(self) -> "NetworkStreamSubscription":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class NetworkStreamHub:
    def __init__(
        self,
        capture_factory: Callable[..., object] | None = None,
        *,
        max_streams: int = 8,
        max_subscribers_per_stream: int = 8,
    ) -> None:
        self._capture_factory = capture_factory or _open_network_capture
        self._max_streams = max(1, int(max_streams))
        self._max_subscribers_per_stream = max(1, int(max_subscribers_per_stream))
        self._lock = threading.Lock()
        self._workers: dict[str, _CaptureWorker] = {}

    def subscribe(self, url: str) -> NetworkStreamSubscription:
        key = normalize_stream_url(url)
        with self._lock:
            worker = self._workers.get(key)
            if worker is None:
                if len(self._workers) >= self._max_streams:
                    raise NetworkStreamError("too many active network camera streams")
                worker = _CaptureWorker(key, self._capture_factory)
                self._workers[key] = worker
            if worker.subscriber_count() >= self._max_subscribers_per_stream:
                raise NetworkStreamError("too many subscribers for network camera stream")
            worker.add_subscriber()
        return NetworkStreamSubscription(self, key, worker)

    def active_stream_count(self) -> int:
        with self._lock:
            return len(self._workers)

    def subscriber_count(self, url: str) -> int:
        key = normalize_stream_url(url)
        with self._lock:
            worker = self._workers.get(key)
            return worker.subscriber_count() if worker is not None else 0

    def _unsubscribe(self, key: str, worker: _CaptureWorker) -> None:
        should_stop = False
        with self._lock:
            current = self._workers.get(key)
            if current is not worker:
                return
            if worker.remove_subscriber() == 0:
                self._workers.pop(key, None)
                should_stop = True
        if should_stop:
            worker.stop()

    def close_all(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop()


def _open_network_capture(url: str, backend: int):
    """Open FFmpeg streams with bounded open/read waits when OpenCV supports it."""
    params: list[int] = []
    open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
    if open_timeout is not None:
        params.extend([int(open_timeout), 8000])
    if read_timeout is not None:
        params.extend([int(read_timeout), 5000])
    if params:
        try:
            return cv2.VideoCapture(url, backend, params)
        except (TypeError, cv2.error):
            pass
    return cv2.VideoCapture(url, backend)


network_stream_hub = NetworkStreamHub()
