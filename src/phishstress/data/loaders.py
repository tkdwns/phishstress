"""데이터 다운로드와 적재.

의존성을 일부러 stdlib로 제한했다. 서빙 컨테이너는 `.[redis]`만 설치하므로
pandas/scikit-learn이 core에 들어가면 이미지가 불필요하게 무거워진다.
28MB CSV는 stdlib `csv`로 충분히 다룬다.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .registry import DatasetSpec, get_spec

# 전사문 한 칸이 매우 길다(평균 4천자, 최대 2만6천자). 기본 필드 상한을 올려둔다.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

_WS = re.compile(r"\s+")


def default_data_dir() -> Path:
    """PHISHSTRESS_DATA_DIR로 덮어쓸 수 있다. 기본은 저장소 루트의 data/ (gitignore됨)."""
    env = os.getenv("PHISHSTRESS_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "data"


@dataclass(frozen=True)
class Example:
    """평가 단위 하나."""

    uid: str
    text: str
    label: int

    @property
    def length(self) -> int:
        return len(self.text)


def normalize_text(raw: str) -> str:
    """공백만 정규화한다. 그 이상은 손대지 않는다 —
    전처리를 강하게 걸면 나중에 적대적 변형의 효과와 구분이 안 된다."""
    return _WS.sub(" ", str(raw)).strip()


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch(spec: DatasetSpec | str, data_dir: Path | None = None, force: bool = False) -> Path:
    """원본 CSV를 내려받고 체크섬을 확인한다. 이미 있으면 재사용한다."""
    spec = get_spec(spec) if isinstance(spec, str) else spec
    data_dir = data_dir or default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / spec.filename

    if target.exists() and not force:
        digest = sha256_of(target)
        if not spec.sha256 or digest == spec.sha256:
            return target
        raise RuntimeError(
            f"{target} 체크섬 불일치.\n  기대: {spec.sha256}\n  실제: {digest}\n"
            "업스트림이 바뀌었거나 파일이 손상됐습니다. force=True로 다시 받으세요."
        )

    tmp = target.with_suffix(".tmp")
    with urllib.request.urlopen(spec.url, timeout=120) as resp, tmp.open("wb") as out:
        while block := resp.read(1 << 20):
            out.write(block)

    digest = sha256_of(tmp)
    if spec.sha256 and digest != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"내려받은 파일의 체크섬이 사양과 다릅니다.\n  기대: {spec.sha256}\n  실제: {digest}"
        )
    tmp.replace(target)
    return target


def load(
    spec: DatasetSpec | str,
    data_dir: Path | None = None,
    drop_duplicates: bool = True,
) -> list[Example]:
    """CSV를 Example 리스트로 읽는다.

    drop_duplicates: 정규화 후 완전히 동일한 전사문을 제거한다. 중복이 train/test에
    갈라져 들어가면 성능이 부풀려지므로 기본값을 True로 둔다.
    """
    spec = get_spec(spec) if isinstance(spec, str) else spec
    path = fetch(spec, data_dir)

    examples: list[Example] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            raw = row.get(spec.text_column)
            lab = row.get(spec.label_column)
            if raw is None or lab is None or str(lab).strip() == "":
                continue
            text = normalize_text(raw)
            if not text:
                continue
            if drop_duplicates:
                key = hashlib.md5(text.encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
            examples.append(
                Example(
                    uid=f"{spec.key}:{i}",
                    text=text,
                    label=int(int(lab) == spec.positive_label),
                )
            )
    if not examples:
        raise RuntimeError(f"{path} 에서 읽어들인 샘플이 없습니다. 열 이름을 확인하세요.")
    return examples


def class_counts(examples: list[Example]) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for e in examples:
        counts[e.label] += 1
    return counts
