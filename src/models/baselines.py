"""기준 모델 4개 (제안서 5.1절, C1 업무).

- GlobalMean : 모든 예측 = train 전체 평균
- UserMean   : 사용자 평균 (train에 없는 사용자는 전역 평균)
- ItemMean   : 상품 평균   (train에 없는 상품은 전역 평균)
- Popularity : Top-N 전용. 점수 = log(1 + train 평점 수). predict는 ItemMean과 같게 둔다

모든 모델은 **정수 번호 배열**(uidx, iidx)을 한꺼번에 받는다 (docs/EVAL_RULES.md 규칙 2).
    predict(uidx, iidx) -> 같은 길이의 예측 평점, 1~5 clip, NaN 없음
    score(uidx, iidx)   -> Top-N용 원점수 (clip 없음)
    score_rows(uidx)    -> (len(uidx), n_items) 행렬. 프로토콜 ③(전체 상품 랭킹)용
문자열 ID 한 명짜리 호출이 필요하면 `IdAdapter`로 감싼다 (interface.py 형식).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import RATING_MIN, RATING_MAX


def _as_idx(df: pd.DataFrame) -> pd.DataFrame:
    """train DataFrame에 uidx·iidx가 없으면 에러. (user_map/item_map 번호를 미리 붙여서 넘긴다)"""
    for c in ['uidx', 'iidx', 'rating']:
        if c not in df.columns:
            raise ValueError(f'train에 {c} 컬럼이 필요합니다 (metrics.attach_index 사용)')
    return df


class _Base:
    name = 'base'

    def fit(self, train: pd.DataFrame, n_users: int, n_items: int):
        train = _as_idx(train)
        self.n_users, self.n_items = int(n_users), int(n_items)
        self.mu = float(train['rating'].mean())
        self._fit(train)
        return self

    def _fit(self, train):
        pass

    # --- 공통 형식 ---
    def predict(self, uidx, iidx) -> np.ndarray:
        p = self._raw(np.asarray(uidx, dtype=np.int64), np.asarray(iidx, dtype=np.int64))
        return np.clip(p, RATING_MIN, RATING_MAX).astype(np.float64)

    def score(self, uidx, iidx) -> np.ndarray:
        return self._raw(np.asarray(uidx, dtype=np.int64), np.asarray(iidx, dtype=np.int64)).astype(np.float64)

    def score_rows(self, uidx) -> np.ndarray:
        """(len(uidx), n_items). 기본 구현은 score를 상품 전체에 대해 부른다."""
        uidx = np.asarray(uidx, dtype=np.int64)
        all_items = np.arange(self.n_items, dtype=np.int64)
        out = np.empty((len(uidx), self.n_items), dtype=np.float64)
        for r, u in enumerate(uidx):
            out[r] = self.score(np.full(self.n_items, u), all_items)
        return out

    def _raw(self, uidx, iidx):
        raise NotImplementedError


class GlobalMean(_Base):
    name = 'global_mean'

    def _raw(self, uidx, iidx):
        return np.full(len(uidx), self.mu)


class UserMean(_Base):
    name = 'user_mean'

    def _fit(self, train):
        s = train.groupby('uidx')['rating'].mean()
        self.user_mean = np.full(self.n_users, self.mu)
        self.user_mean[s.index.values] = s.values

    def _raw(self, uidx, iidx):
        return self.user_mean[uidx]

    def score_rows(self, uidx):
        uidx = np.asarray(uidx, dtype=np.int64)
        return np.repeat(self.user_mean[uidx][:, None], self.n_items, axis=1)


class _ItemVector(_Base):
    """점수가 상품에만 의존하는 모델의 공통 부분."""
    item_vec: np.ndarray

    def _raw(self, uidx, iidx):
        return self.item_vec[iidx]

    def score_rows(self, uidx):
        return np.repeat(self.item_vec[None, :], len(uidx), axis=0)


class ItemMean(_ItemVector):
    name = 'item_mean'

    def _fit(self, train):
        s = train.groupby('iidx')['rating'].mean()
        self.item_vec = np.full(self.n_items, self.mu)
        self.item_vec[s.index.values] = s.values


class Popularity(_ItemVector):
    """Top-N 기준 모델. score = log(1 + train 평점 수). predict는 상품 평균(평점 트랙 비교용)."""
    name = 'popularity'

    def _fit(self, train):
        cnt = np.bincount(train['iidx'].values, minlength=self.n_items)
        self.item_vec = np.log1p(cnt).astype(np.float64)
        s = train.groupby('iidx')['rating'].mean()
        self.item_mean = np.full(self.n_items, self.mu)
        self.item_mean[s.index.values] = s.values

    def predict(self, uidx, iidx):
        return np.clip(self.item_mean[np.asarray(iidx, dtype=np.int64)], RATING_MIN, RATING_MAX)


BASELINES = {m.name: m for m in [GlobalMean, UserMean, ItemMean, Popularity]}


class IdAdapter:
    """interface.py 형식(predict(user_id, item_ids), 문자열 ID 한 명씩)이 필요할 때 감싼다."""

    def __init__(self, model, user_map: pd.DataFrame, item_map: pd.DataFrame):
        self.m = model
        self.u = pd.Series(user_map['uidx'].values, index=user_map['user_id'].astype(str).values)
        self.i = pd.Series(item_map['iidx'].values, index=item_map['item_id'].astype(str).values)

    def _idx(self, user_id, item_ids):
        uu = int(self.u[str(user_id)])
        ii = self.i.reindex(pd.Index(item_ids).astype(str)).values
        if np.isnan(ii).any():
            raise ValueError('item_map에 없는 상품 ID')
        return np.full(len(ii), uu), ii.astype(np.int64)

    def predict(self, user_id, item_ids):
        return self.m.predict(*self._idx(user_id, item_ids))

    def score(self, user_id, item_ids):
        return self.m.score(*self._idx(user_id, item_ids))
