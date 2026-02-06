from __future__ import annotations

from dataclasses import dataclass, field

from .analyzer import Thresholds


@dataclass
class SessionState:
    thresholds: Thresholds = field(default_factory=Thresholds)
    total_nm: list[float] = field(default_factory=list)

    def add_measurements(self, values: list[float]) -> None:
        self.total_nm.extend(values)

    def total_count(self) -> int:
        return len(self.total_nm)
