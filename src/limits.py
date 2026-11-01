import os
import time
from collections import deque


def demo_mode():
    return os.getenv("DEMO_MODE", "").strip().lower() in ("1", "true", "yes")


def ai_limit_per_hour():
    try:
        return max(1, int(os.getenv("AI_REQUESTS_PER_HOUR", "30")))
    except ValueError:
        return 30


class HourlyBudget:
    def __init__(self):
        self.calls = deque()

    def try_spend(self, limit, now=None):
        now = time.monotonic() if now is None else now
        while self.calls and now - self.calls[0] >= 3600:
            self.calls.popleft()
        if len(self.calls) >= limit:
            return False
        self.calls.append(now)
        return True

    def clear(self):
        self.calls.clear()


ai_budget = HourlyBudget()