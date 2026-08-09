from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RequestDeadline:
    """Monotonic absolute deadline shared across every provider stage."""

    expires_at: float
    _clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.expires_at) is not float
            or not math.isfinite(self.expires_at)
            or not callable(self._clock)
        ):
            raise ValueError("request deadline is invalid")

    @classmethod
    def after(
        cls,
        seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> RequestDeadline:
        if type(seconds) not in {float, int} or isinstance(seconds, bool):
            raise ValueError("request deadline duration must be numeric")
        duration = float(seconds)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("request deadline duration must be positive and finite")
        return cls(expires_at=float(clock()) + duration, _clock=clock)

    def remaining_seconds(self) -> float:
        return max(0.0, self.expires_at - float(self._clock()))

    def raise_if_expired(self) -> None:
        if self.remaining_seconds() <= 0:
            raise TimeoutError("research request deadline expired")

    def child(self, maximum_seconds: float) -> RequestDeadline:
        if type(maximum_seconds) not in {float, int} or isinstance(maximum_seconds, bool):
            raise ValueError("child deadline duration must be numeric")
        duration = float(maximum_seconds)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("child deadline duration must be positive and finite")
        self.raise_if_expired()
        return RequestDeadline(
            expires_at=min(self.expires_at, float(self._clock()) + duration),
            _clock=self._clock,
        )
