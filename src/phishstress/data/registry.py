"""데이터셋 사양.

원본 데이터는 저장소에 넣지 않는다(docs/ETHICS.md). 대신 **어디서 무엇을 받아
무결성을 어떻게 확인하는지**를 코드로 고정한다. 재현성은 파일을 커밋해서가 아니라
URL + 체크섬으로 확보한다.

체크섬을 박아두는 이유: 업스트림이 조용히 파일을 바꾸면 우리 실험 수치가 아무 경고
없이 달라진다. 그 순간 실패하게 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    url: str
    sha256: str
    text_column: str
    label_column: str
    positive_label: int
    description: str
    citation: str

    @property
    def filename(self) -> str:
        return f"{self.key}.csv"


_RAW = "https://raw.githubusercontent.com/selfcontrol7/Korean_Voice_Phishing_Detection/main"

DATASETS: dict[str, DatasetSpec] = {
    # Mathematics 11(14), 3217 (2023)이 정확도 99.32% / F1 99.31%를 보고할 때 쓴 바로 그 파일.
    # 우리가 이 데이터를 고른 이유는 성능을 겨루기 위해서가 아니라, 그 수치가 어떻게
    # 만들어졌는지 확인하기 위해서다. docs/EXPERIMENTS.md의 길이 아티팩트 분석 참조.
    "korccvi_v2": DatasetSpec(
        key="korccvi_v2",
        url=f"{_RAW}/Data_Collection_Preprocessing/KorCCVi_v2.csv",
        sha256="0795574b086f73f2757c2867960f2c6cc268d9a26331a04e861dc3721ef01d20",
        text_column="transcript",
        label_column="label",
        positive_label=1,
        description="Korean Call Content Vishing v2 — 통화 전사 텍스트, 피싱/정상 이진 라벨",
        citation="Mathematics 11(14), 3217 (2023); KorCCVi v2",
    ),
    # 균형 데이터(609/609). v2와 길이 아티팩트의 방향이 반대라 교차 확인에 쓴다.
    "korccvid_v13": DatasetSpec(
        key="korccvid_v13",
        url=f"{_RAW}/KoBERT/KorCCViD_v1.3_fullcleansed.csv",
        sha256="23caceb385194d5694388722bc5f15e1e09c2cfb5585264bc8b94b2c46ae494a",
        text_column="Transcript",
        label_column="Label",
        positive_label=1,
        description="KorCCViD v1.3 fullcleansed — 균형 이진 라벨(609/609)",
        citation="KorCCViD v1.3",
    ),
}


def get_spec(key: str) -> DatasetSpec:
    if key not in DATASETS:
        raise KeyError(f"알 수 없는 데이터셋: {key}. 사용 가능: {sorted(DATASETS)}")
    return DATASETS[key]
