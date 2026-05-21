import sys
import threading
from collections import deque
from datetime import datetime
from typing import List


class DevLogBuffer:
    def __init__(self, max_lines: int = 300):
        self._lines = deque(maxlen=max_lines)
        self._lock = threading.Lock()

    def append(self, text: str) -> None:
        for line in text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            if "/api/dev/logs" in clean:
                continue
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            with self._lock:
                self._lines.append(f"{timestamp} {clean}")

    def tail(self, limit: int = 120) -> List[str]:
        safe_limit = max(1, min(limit, self._lines.maxlen or 300))
        with self._lock:
            return list(self._lines)[-safe_limit:]


class TeeStdout:
    def __init__(self, original, buffer: DevLogBuffer):
        self._original = original
        self._buffer = buffer
        self._pending = ""

    def write(self, text: str):
        written = self._original.write(text)
        self._pending += text
        if "\n" in self._pending:
            parts = self._pending.split("\n")
            for line in parts[:-1]:
                self._buffer.append(line)
            self._pending = parts[-1]
        return written

    def flush(self):
        if self._pending.strip():
            self._buffer.append(self._pending)
            self._pending = ""
        return self._original.flush()

    def isatty(self):
        return self._original.isatty()

    def __getattr__(self, name: str):
        return getattr(self._original, name)


dev_log_buffer = DevLogBuffer()


def install_stdout_capture() -> None:
    if isinstance(sys.stdout, TeeStdout):
        return
    sys.stdout = TeeStdout(sys.stdout, dev_log_buffer)
    dev_log_buffer.append("AI dev stdout log capture attached")
