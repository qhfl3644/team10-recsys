"""모든 추천 모델이 따르는 공통 형식 (10/24 킥오프 합의안).

하이브리드·평가 코드는 모델 종류와 상관없이 아래 두 메서드만 호출한다.
"""
from typing import Protocol, Sequence

import numpy as np


class Recommender(Protocol):
    def fit(self, train) -> "Recommender":
        """train: DataFrame (user_id, item_id, rating, timestamp)"""
        ...

    def predict(self, user_id: str, item_ids: Sequence[str]) -> np.ndarray:
        """평점 예측 트랙용.
        - item_ids와 같은 순서·같은 길이의 float 배열
        - 1~5로 clip
        - NaN 금지: 모르는 사용자·상품은 대체값(전역 평균 등)
        """
        ...

    def score(self, user_id: str, item_ids: Sequence[str]) -> np.ndarray:
        """Top-N 트랙용 원점수. clip하지 않는다 (5점 근처 상품끼리 순서 차이를 보존).
        모델에 별도 점수가 없으면 predict와 같은 값을 clip 없이 반환."""
        ...


# 예측값 파일 형식 (config.pred_path 위치에 저장)
PRED_COLUMNS = ['user_id', 'item_id', 'pred']
