import time


class Stopwatch:
    def __init__(self):
        self.start_time = None
        self.last_lap = None

    def start(self):
        self.start_time = self.last_lap = time.time()

    def lap(self):
        now = time.time()
        elapsed = now - self.last_lap
        self.last_lap = now
        return elapsed

    def elapsed(self):
        return time.time() - self.last_lap

    def total_elapsed(self):
        return time.time() - self.start_time


def fmt_sec(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.3f}s"

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    secs = seconds % 60

    if seconds < 3600:
        return f"{minutes}m{int(secs)}s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours}h{minutes}m{secs}s"


def flint(num):
    return int(num) if float(num).is_integer() else float(num)
