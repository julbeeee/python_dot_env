from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CsvSaveResult:
    path: str
    count: int


def save_csv(values: list[float], path: str) -> CsvSaveResult:
    with open(path, "w", encoding="utf-8") as f:
        for v in values:
            f.write(f"{v}\n")
    return CsvSaveResult(path=path, count=len(values))
