"""가짜 프로필 주입 (D1~D2)

표준 공격 모델 세 가지. 목표 상품에는 항상 최고 평점을 준다.
    random    : 채움 평점을 전역 평점 분포(평균·표준편차)에서 뽑는다.
    average   : 채움 평점을 각 상품의 평균 근처에서 뽑는다 (탐지가 가장 어렵다)
    bandwagon : 인기 상품 몇 개에 최고 평점을 주고 나머지는 random과 같다.

프로필 길이
    기본값은 실제 사용자의 train 평점 수 분포에서 뽑는다. 길이를 한 값으로 고정하면 탐지기가 공격 행동이 아니라 "길이가 똑같다"는 것만 학습해 F1이 비현실적으로 높아진다.
    n_filler에 정수를 주면 고정 길이로도 쓸 수 있다.
    어느 쪽이든 프로필의 총 평점 수 = 채움 상품 수 + 목표 상품 1개다.

모든 함수는 정수 번호(uidx, iidx)로 동작
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import RATING_MIN, RATING_MAX

KINDS = ['random', 'average', 'bandwagon']


def item_stats(train: pd.DataFrame, n_items: int):
    """상품별 train 평점 평균·표준편차·개수 (iidx 순) + 전역 평균·표준편차"""
    g = train.groupby('iidx')['rating']
    mu = float(train['rating'].mean())
    sd = float(train['rating'].std())
    mean = g.mean().reindex(range(n_items)).to_numpy()
    std = g.std().reindex(range(n_items)).fillna(0.0).to_numpy()
    count = g.size().reindex(range(n_items), fill_value=0).to_numpy()
    mean = np.where(np.isnan(mean), mu, mean)
    return mean, std, count, mu, sd


def user_length_pool(train: pd.DataFrame) -> np.ndarray:
    """실제 사용자의 train 평점 수 배열. 가짜 프로필 길이를 여기서 뽑는다."""
    return train.groupby('uidx').size().to_numpy()


def pick_targets(count: np.ndarray, min_count: int = 5) -> dict:
    """공격 타깃 두 개. 인기 = train 평점 수 최대, 비인기 = min_count 이상 중 최소"""
    order = np.argsort(-count)
    eligible = order[count[order] >= min_count]
    if len(eligible) == 0:
        raise ValueError(f'train 평점 {min_count}개 이상인 상품이 없습니다')
    return {'popular': int(eligible[0]), 'unpopular': int(eligible[-1])}


def inject(train: pd.DataFrame, n_users: int, n_items: int, target: int, kind: str,
           n_fake: int, stats, rng, n_filler=None, length_pool=None, n_bandwagon: int = 5):
    """가짜 프로필을 만들어 train에 덧붙인다.

    n_filler : 정수면 채움 상품 수 고정. None이면 length_pool에서 뽑는다.
    반환: (늘어난 train, 가짜 사용자 uidx 배열)
    """
    if kind not in KINDS:
        raise ValueError(f'kind는 {KINDS} 중 하나여야 합니다')
    if n_filler is None and length_pool is None:
        raise ValueError('n_filler 또는 length_pool 중 하나는 필요합니다')

    mean, std, count, mu, sd = stats
    popular = np.argsort(-count)[:50]
    popular = popular[popular != target]
    all_items = np.arange(n_items)

    rows = []
    for f in range(n_fake):
        total = int(n_filler) + 1 if n_filler is not None else int(rng.choice(length_pool))
        total = max(total, 2)                       # 목표 1개 + 채움 최소 1개

        items = [target]
        ratings = [RATING_MAX]

        if kind == 'bandwagon':
            take = min(n_bandwagon, len(popular), total - 1)
            sel = rng.choice(popular, size=take, replace=False)
            items += list(sel)
            ratings += [RATING_MAX] * take

        # 이미 고른 상품을 뺀 풀에서 한 번에 뽑는다 (겹쳐서 길이가 줄지 않게)
        need = total - len(items)
        if need > 0:
            pool = np.setdiff1d(all_items, np.asarray(items))
            fill = rng.choice(pool, size=min(need, len(pool)), replace=False)
            v = (rng.normal(mean[fill], np.maximum(std[fill], 0.3)) if kind == 'average'
                 else rng.normal(mu, sd, len(fill)))
            items += list(fill)
            ratings += list(np.clip(np.round(v), RATING_MIN, RATING_MAX))

        rows.append(pd.DataFrame({'uidx': np.int32(n_users + f),
                                  'iidx': np.asarray(items, dtype=np.int32),
                                  'rating': np.asarray(ratings, dtype=np.float64)}))

    fake = pd.concat(rows, ignore_index=True)
    cols = ['uidx', 'iidx', 'rating']
    return (pd.concat([train[cols], fake[cols]], ignore_index=True),
            np.arange(n_users, n_users + n_fake, dtype=np.int64))

