"""벤치마크 진단 — "이 데이터셋은 얼마나 쉬운가?"

모델 성능을 재기 전에 **데이터셋 자체를 감사**한다. Day 3에 이 진단을 돌린 결과가
프로젝트의 방향을 바꿨다.

KorCCVi v2에서 발견한 것:

  1. 길이 아티팩트 — 글자 수만 세도 정확도 96.3%. 두 클래스가 다른 출처에서
     수집되어 정상 통화는 최소 1,740자인데 피싱 전사문 중앙값은 534자다.
  2. 주제 분리 — 정상 클래스가 은행 통화가 아니라 **여행·반려동물·음식 잡담**이다.
     일상 어휘 포함률이 정상 95.1% vs 피싱 14.6%, 금융 어휘는 피싱 87.6% vs 정상 19.3%.

즉 이 벤치마크가 실제로 요구하는 과제는 "은행 사기 대본과 반려동물 수다를 구별하라"이다.
배포 환경의 어려운 음성 샘플 — 진짜 은행 상담, 실제 대출 안내, 진짜 경찰 통화 — 은
데이터셋에 아예 없다. 선행 연구가 보고한 99%대 수치는 여기서 나온다.

이 모듈은 그 판단을 눈대중이 아니라 수치로 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data.loaders import Example

# 금융·수사 어휘. 금융감독원·경찰청 공개 안내문 수준의 일반 명사만 쓴다.
FINANCIAL_VOCAB: tuple[str, ...] = (
    "계좌",
    "이체",
    "송금",
    "대출",
    "금융",
    "은행",
    "입금",
    "카드",
    "금리",
    "검찰",
    "경찰",
    "수사",
    "명의",
    "개인정보",
)

# 일상 대화 어휘.
CASUAL_VOCAB: tuple[str, ...] = (
    "여행",
    "친구",
    "영화",
    "음식",
    "학교",
    "가족",
    "취미",
    "반려",
    "드라마",
    "커피",
    "방학",
    "게임",
    "날씨",
    "운동",
)


def rank_auc(values: list[float], labels: list[int]) -> float:
    """단일 실수 특징이 라벨을 얼마나 설명하는지 (= Mann-Whitney U / ROC-AUC).

    0.5면 무정보, 0 또는 1이면 그 특징 하나로 완벽히 갈린다.
    """
    pairs = sorted(zip(values, labels, strict=True))
    n = len(pairs)
    if n == 0:
        return float("nan")
    # 동점 평균 순위 부여
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos_ranks = sum(r for r, (_, y) in zip(ranks, pairs, strict=True) if y == 1)
    n_pos = sum(y for _, y in pairs)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    u = pos_ranks - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def vocab_hit_rate(examples: list[Example], vocab: tuple[str, ...], label: int) -> float:
    """해당 클래스에서 어휘 목록 중 하나라도 등장하는 샘플의 비율."""
    group = [e for e in examples if e.label == label]
    if not group:
        return float("nan")
    return sum(1 for e in group if any(w in e.text for w in vocab)) / len(group)


@dataclass
class DatasetDiagnosis:
    n: int
    positives: int
    positive_rate: float
    length_auc: float
    length_ceiling_accuracy: float
    financial_hit_positive: float
    financial_hit_negative: float
    casual_hit_positive: float
    casual_hit_negative: float
    warnings: list[str] = field(default_factory=list)

    @property
    def length_separability(self) -> float:
        """0.5에서 얼마나 벗어났는가. 0이면 길이가 무정보."""
        return abs(self.length_auc - 0.5) * 2

    @property
    def topical_separability(self) -> float:
        """어휘군 두 개의 클래스 간 비율 차이 평균. 1에 가까울수록 주제가 갈려 있다."""
        fin = abs(self.financial_hit_positive - self.financial_hit_negative)
        cas = abs(self.casual_hit_positive - self.casual_hit_negative)
        return (fin + cas) / 2

    def to_dict(self) -> dict:
        out = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in self.__dict__.items()
            if k != "warnings"
        }
        out["length_separability"] = round(self.length_separability, 4)
        out["topical_separability"] = round(self.topical_separability, 4)
        out["warnings"] = list(self.warnings)
        return out


def _length_ceiling(examples: list[Example]) -> float:
    """길이 임계값 하나로 달성 가능한 최고 정확도."""
    pts = sorted((e.length, e.label) for e in examples)
    n = len(pts)
    total_pos = sum(y for _, y in pts)
    best = max(total_pos, n - total_pos) / n  # 다수클래스
    # "임계값 미만이면 양성" 방향
    pos_left = 0
    for i, (_, y) in enumerate(pts, start=1):
        pos_left += y
        correct = pos_left + (n - i) - (total_pos - pos_left)
        best = max(best, correct / n, (n - correct) / n)
    return best


def diagnose(examples: list[Example]) -> DatasetDiagnosis:
    """데이터셋을 감사하고 경고를 붙인다."""
    if not examples:
        raise ValueError("빈 데이터셋입니다.")
    labels = [e.label for e in examples]
    n_pos = sum(labels)
    if n_pos in (0, len(examples)):
        raise ValueError("한 클래스만 있어 진단할 수 없습니다.")

    d = DatasetDiagnosis(
        n=len(examples),
        positives=n_pos,
        positive_rate=n_pos / len(examples),
        length_auc=rank_auc([float(e.length) for e in examples], labels),
        length_ceiling_accuracy=_length_ceiling(examples),
        financial_hit_positive=vocab_hit_rate(examples, FINANCIAL_VOCAB, 1),
        financial_hit_negative=vocab_hit_rate(examples, FINANCIAL_VOCAB, 0),
        casual_hit_positive=vocab_hit_rate(examples, CASUAL_VOCAB, 1),
        casual_hit_negative=vocab_hit_rate(examples, CASUAL_VOCAB, 0),
    )

    majority = max(d.positive_rate, 1 - d.positive_rate)
    if d.length_ceiling_accuracy > majority + 0.10:
        d.warnings.append(
            f"길이 아티팩트: 글자 수 임계값 하나로 정확도 {d.length_ceiling_accuracy:.3f} "
            f"(다수클래스 {majority:.3f}). 이 위에서 잰 성능은 판별 능력이 아닐 수 있다."
        )
    if d.length_separability > 0.5:
        d.warnings.append(
            f"길이 단독 AUC {d.length_auc:.3f} — 두 클래스의 길이 분포가 크게 갈려 있다."
        )
    if abs(d.casual_hit_negative - d.casual_hit_positive) > 0.5:
        d.warnings.append(
            f"주제 분리: 일상 어휘 포함률이 정상 {d.casual_hit_negative:.3f} vs "
            f"피싱 {d.casual_hit_positive:.3f}. 음성 클래스가 금융 통화가 아니라 "
            f"잡담일 가능성이 높다 — 배포 환경의 어려운 음성 샘플이 빠져 있다."
        )
    if d.financial_hit_negative < 0.30 < d.financial_hit_positive:
        d.warnings.append(
            f"하드 네거티브 부재: 금융 어휘를 포함한 정상 샘플이 {d.financial_hit_negative:.3f}뿐. "
            f"진짜 은행 상담·대출 안내가 학습·평가에 들어 있지 않다."
        )
    return d
