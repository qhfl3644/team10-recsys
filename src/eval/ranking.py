"""동점을 무작위로 처리하는 정답 순위 계산 (제안서 2.1절).

평점이 5점에 몰려 있어 예측 점수 동점이 자주 생긴다. 동점을 정답에 유리하게
처리하면(예: 정렬했을 때 정답이 앞에 오도록) HR·NDCG가 부풀려진다.
여기서는 정답 순위 = 1 + (정답보다 점수가 높은 후보 수) + U{0, ..., 동점 후보 수}로 계산한다.
동점끼리 무작위로 섞어 순위를 매기는 것과 분포가 같다.
"""
from __future__ import annotations

import numpy as np


def target_rank(scores: np.ndarray, is_target: np.ndarray, rng) -> np.ndarray:
    """프로토콜 ①②용. scores, is_target: (사용자 수, 후보 수). 행마다 정답이 정확히 1개.
    반환: 사용자별 정답 순위 (1부터)."""
    scores = np.asarray(scores, dtype=np.float64)
    is_target = np.asarray(is_target, dtype=bool)
    if np.isnan(scores).any():
        raise ValueError('점수에 NaN이 있습니다')
    if not (is_target.sum(axis=1) == 1).all():
        raise ValueError('행마다 정답이 정확히 1개여야 합니다')
    t = scores[is_target][:, None]
    higher = (scores > t).sum(axis=1)
    ties = (scores == t).sum(axis=1) - 1
    return 1 + higher + rng.integers(0, ties + 1)


def target_rank_full(score_all: np.ndarray, target: int, exclude: np.ndarray, rng) -> int:
    """프로토콜 ③용. score_all: 전체 상품 점수 (iidx 순서).
    exclude: 순위에서 뺄 상품(정답 제외). sampling.protocol3_exclude 결과를 넣는다."""
    s_t = score_all[target]
    ex = score_all[exclude]
    higher = int((score_all > s_t).sum() - (ex > s_t).sum())
    ties = int((score_all == s_t).sum() - 1 - (ex == s_t).sum())
    return 1 + higher + int(rng.integers(0, ties + 1))


def hr_ndcg_at_k(ranks: np.ndarray, k: int = 10):
    """점검용. 최종 지표 함수는 평가 담당 코드를 쓴다."""
    ranks = np.asarray(ranks)
    hit = ranks <= k
    ndcg = np.where(hit, 1.0 / np.log2(ranks + 1), 0.0)
    return float(hit.mean()), float(ndcg.mean())
