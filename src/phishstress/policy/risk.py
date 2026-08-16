"""위험 등급 정책 레이어.

v1 과제정의서의 '학습형 Fusion'은 폐기했다. 이유는 docs/RED_TEAM_REVIEW.md C-3 참조:
음성 위조 라벨과 사기 대화 라벨이 **동시에 붙은 데이터가 존재하지 않아** 융합 가중치를
학습할 방법이 없다. 선행 연구(Applied Sciences 15(20), 11170)조차 0.8/0.2로 손으로 정했다.

대신 명시적 규칙 정책으로 재정의하고, 그 한계를 문서에 남긴다. 규칙은 넷이다.

  1. 신뢰도 가중 결합 — 판별기가 스스로 보고한 confidence로 발언권을 조절
  2. 열화 인지 감쇠  — 입력이 열화됐다고 판단되면 해당 판별기 가중치를 더 줄인다
  3. EWMA 시간 누적  — 청크 단위 순간 오탐에 등급이 흔들리지 않게
  4. 히스테리시스    — 진입/해제 임계값을 분리해 등급 깜빡임(flapping) 방지

이 중 2번이 선행 연구의 고정 가중치가 못 하는 부분이며, 스트레스 스위트가 있어야만
검증할 수 있다. 벤치마크와 서빙이 서로를 필요로 하는 구조가 여기서 완성된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..detectors.base import DetectionResult


class RiskGrade(str, Enum):
    SAFE = "SAFE"
    WARN = "WARN"
    BLOCK = "BLOCK"

    @property
    def rank(self) -> int:
        return {"SAFE": 0, "WARN": 1, "BLOCK": 2}[self.value]


@dataclass(frozen=True)
class PolicyConfig:
    """임계값은 Day 11에 비용 곡선(미탐:오탐 = 20:1 가정)으로 재산출한다.

    현재 값은 Day 1 스텁용 자리표시자이며, README에 그렇게 명시한다.
    """

    alpha: float = 0.3  # EWMA 평활 계수. 클수록 최근 청크에 민감
    warn_enter: float = 0.60
    warn_exit: float = 0.45
    block_enter: float = 0.85
    block_exit: float = 0.70
    min_confidence_floor: float = 0.05  # 가중치가 0이 되어 판별기가 사라지는 것 방지
    base_weights: dict[str, float] = field(default_factory=lambda: {"text": 0.7, "audio": 0.3})

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha는 (0,1] 범위여야 합니다.")
        if not self.warn_exit < self.warn_enter:
            raise ValueError("warn_exit < warn_enter 여야 히스테리시스가 성립합니다.")
        if not self.block_exit < self.block_enter:
            raise ValueError("block_exit < block_enter 여야 히스테리시스가 성립합니다.")
        if not self.warn_enter < self.block_enter:
            raise ValueError("warn_enter < block_enter 여야 합니다.")


@dataclass(frozen=True)
class PolicyState:
    """정책 1회 갱신 결과. 그대로 WebSocket 응답으로 직렬화된다."""

    grade: RiskGrade
    risk: float  # EWMA 누적 위험도
    instant_risk: float  # 이번 청크의 결합 위험도 (누적 전)
    updates: int
    contributions: dict[str, dict[str, float]]
    degraded: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "grade": self.grade.value,
            "risk": round(self.risk, 4),
            "instant_risk": round(self.instant_risk, 4),
            "updates": self.updates,
            "degraded": self.degraded,
            "contributions": {
                k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in self.contributions.items()
            },
        }


class RiskPolicy:
    """세션 하나의 위험 상태를 들고 있는 정책 엔진.

    상태는 (EWMA 위험도, 현재 등급, 갱신 횟수) 세 개뿐이라 직렬화가 쉽다 —
    Redis에 넣어 재접속 시 복원할 수 있고, 이것이 SessionStore가 저장하는 값이다.
    """

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()
        self._risk: float | None = None
        self._grade = RiskGrade.SAFE
        self._updates = 0

    # ---- 상태 직렬화 (Redis 복원용) --------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {"risk": self._risk, "grade": self._grade.value, "updates": self._updates}

    def restore(self, snapshot: dict[str, Any]) -> None:
        self._risk = snapshot.get("risk")
        self._grade = RiskGrade(snapshot.get("grade", "SAFE"))
        self._updates = int(snapshot.get("updates", 0))

    @property
    def grade(self) -> RiskGrade:
        return self._grade

    @property
    def risk(self) -> float:
        return 0.0 if self._risk is None else self._risk

    # ---- 핵심 동작 -------------------------------------------------------

    def update(self, results: dict[str, DetectionResult]) -> PolicyState:
        """판별기별 결과를 받아 등급을 갱신한다.

        results: {modality: DetectionResult}. 예: {"text": ..., "audio": ...}
                 이번 청크에서 얻지 못한 modality는 빼고 넘기면 된다.
        """
        cfg = self.config
        contributions: dict[str, dict[str, float]] = {}
        weighted_sum = 0.0
        weight_total = 0.0
        degraded = False

        for modality, result in results.items():
            base = cfg.base_weights.get(modality, 0.0)
            if base <= 0.0:
                continue
            # 규칙 1+2: 기본 가중치 × 판별기 자기신뢰도
            weight = max(base * result.confidence, cfg.min_confidence_floor * base)
            if result.confidence < 0.5:
                degraded = True
            weighted_sum += weight * result.score
            weight_total += weight
            contributions[modality] = {
                "score": result.score,
                "confidence": result.confidence,
                "effective_weight": weight,
            }

        instant = weighted_sum / weight_total if weight_total > 0 else 0.0

        # 규칙 3: EWMA 누적
        if self._risk is None:
            self._risk = instant
        else:
            self._risk = cfg.alpha * instant + (1.0 - cfg.alpha) * self._risk

        self._updates += 1
        self._grade = self._apply_hysteresis(self._risk, self._grade)

        # 가중치를 상대 기여도로 정규화해 응답에 담는다
        if weight_total > 0:
            for c in contributions.values():
                c["effective_weight"] = c["effective_weight"] / weight_total

        return PolicyState(
            grade=self._grade,
            risk=self._risk,
            instant_risk=instant,
            updates=self._updates,
            contributions=contributions,
            degraded=degraded,
        )

    def _apply_hysteresis(self, risk: float, current: RiskGrade) -> RiskGrade:
        """규칙 4: 진입 임계값은 높게, 해제 임계값은 낮게 두어 등급 깜빡임을 막는다."""
        cfg = self.config
        if current is RiskGrade.BLOCK:
            if risk < cfg.block_exit:
                return RiskGrade.WARN if risk >= cfg.warn_exit else RiskGrade.SAFE
            return RiskGrade.BLOCK

        if current is RiskGrade.WARN:
            if risk >= cfg.block_enter:
                return RiskGrade.BLOCK
            if risk < cfg.warn_exit:
                return RiskGrade.SAFE
            return RiskGrade.WARN

        # SAFE
        if risk >= cfg.block_enter:
            return RiskGrade.BLOCK
        if risk >= cfg.warn_enter:
            return RiskGrade.WARN
        return RiskGrade.SAFE

    def reset(self) -> None:
        self._risk = None
        self._grade = RiskGrade.SAFE
        self._updates = 0
