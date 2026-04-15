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


class Glyphs:
    dH = chr(0x2550)   # ═
    dV = chr(0x2551)   # ║
    dDR = chr(0x2554)  # ╔
    dDL = chr(0x2557)  # ╗
    dUR = chr(0x255A)  # ╚
    dUL = chr(0x255D)  # ╝
    dVR = chr(0x2560)  # ╠
    dVL = chr(0x2563)  # ╣
    dHD = chr(0x2566)  # ╦
    dHU = chr(0x2569)  # ╩
    d4 = chr(0x256C)   # ╬
    bul = chr(0x2022)  # •
