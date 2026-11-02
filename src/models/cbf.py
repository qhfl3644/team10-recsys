"""콘텐츠 기반 필터링 (B4, 제안서 3.2절)

상품 텍스트(item_features)를 TF-IDF로 벡터화하고 L2 정규화한다. 두 벡터의 내적 = 코사인 유사도.
예측은 강의 module 6의 아이템 기반 CF 공식에 평점 유사도 대신 콘텐츠 유사도를 넣는다.

    r_hat(u, i) = sum_{j in N_k(i;u)} sim(i, j) * r_uj  /  sum_{j in N_k(i;u)} sim(i, j)

- N_k(i;u): 사용자 u가 train에서 평가한 상품 중 i와 가장 비슷한 k개 (k=None이면 전부)
- 분모가 0이면(비슷한 상품이 없으면) 사용자 평균 평점
- 사용자 이력은 train만 쓴다 (MF와 같은 조건)

Top-N 트랙용 원점수 (score, score_rows):
    s(u, i) = 사용자가 train에서 평가한 상품과의 유사도 중 상위 score_k개의 평균 (score_k=1이면 최댓값)
    평점을 곱하지 않은 순수 유사도라 5점 쏠림의 영향을 받지 않는다.

레포 공통 형식(docs/EVAL_RULES.md 규칙 2)을 따른다.
    predict(uidx, iidx)  예측 평점, 1~5 clip
    score(uidx, iidx)    Top-N 원점수 (유사도), clip 없음
    score_rows(uidx)     (len(uidx), n_items) 원점수 행렬 (프로토콜 ③)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from config import RATING_MIN, RATING_MAX
from src.models.baselines import _Base

TFIDF_DEFAULT = dict(stop_words='english', min_df=3, max_df=0.5)


# =============================================================== TF-IDF
def build_tfidf(texts, stop_words='english', min_df=3, max_df=0.5, max_features=None):
    """상품 텍스트 → (n_items × 단어 수) L2 정규화 TF-IDF 희소 행렬(float32)과 vectorizer."""
    vec = TfidfVectorizer(stop_words=stop_words, min_df=min_df, max_df=max_df,
                          max_features=max_features, dtype=np.float32)
    X = vec.fit_transform(pd.Series(texts).fillna('').astype(str).values)
    X = normalize(X, norm='l2', copy=False)          # 행 길이 1 → 내적이 코사인 유사도
    return sp.csr_matrix(X, dtype=np.float32), vec


def save_tfidf(path_npz: str, X, vec, params: dict):
    sp.save_npz(path_npz, X)
    meta = dict(params=params, shape=list(X.shape), nnz=int(X.nnz),
                vocab_size=len(vec.vocabulary_) if vec is not None else None)
    with open(path_npz.replace('.npz', '_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


def load_tfidf(path_npz: str):
    return sp.csr_matrix(sp.load_npz(path_npz), dtype=np.float32)


def similar_items(X, i: int, n: int = 5):
    """상품 i와 가장 비슷한 상품 n개 (자기 자신 제외). 반환: (iidx 배열, 유사도 배열)"""
    s = np.asarray((X @ X[i].T).todense()).ravel()
    s[i] = -1
    top = np.argpartition(-s, n)[:n]
    top = top[np.argsort(-s[top])]
    return top, s[top]


# =============================================================== CBF 모델
class CBF(_Base):
    name = 'cbf'

    def __init__(self, X, k: int | None = 5, score_k: int | None = 5,
                 pair_budget: int = 200_000, rows_budget: int = 60_000_000):
        """X: (n_items × 단어 수) L2 정규화 TF-IDF (build_tfidf 결과)
        k: 평점 예측에 쓰는 이웃 수 (None = 평가한 상품 전부)
        score_k: Top-N 원점수에 쓰는 상위 유사도 개수 (1 = 최댓값, None = 전부의 평균)
        pair_budget: 한 번에 계산하는 (행, 이력 상품) 쌍 수 상한
        rows_budget: score_rows에서 한 번에 만드는 (상품 × 이력) 밀집 행렬 원소 수 상한"""
        self.X = sp.csr_matrix(X, dtype=np.float32)
        self.k = k
        self.score_k = score_k
        self.pair_budget = int(pair_budget)
        self.rows_budget = int(rows_budget)

    # ------------------------------------------------------------ 학습 = 사용자 이력 정리
    def fit(self, train: pd.DataFrame, n_users: int, n_items: int):
        for c in ['uidx', 'iidx', 'rating']:
            if c not in train.columns:
                raise ValueError(f'train에 {c} 컬럼이 필요합니다 (metrics.attach_index 사용)')
        if self.X.shape[0] != n_items:
            raise ValueError(f'TF-IDF 행 수 {self.X.shape[0]} ≠ 상품 수 {n_items} (item_map 순서 확인)')
        self.n_users, self.n_items = int(n_users), int(n_items)
        self.mu = float(train['rating'].mean())
        t = train.sort_values(['uidx', 'iidx'], kind='stable')
        cnt = np.bincount(t['uidx'].values, minlength=self.n_users)
        self.indptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
        self.h_items = t['iidx'].values.astype(np.int64)
        self.h_ratings = t['rating'].values.astype(np.float64)
        s = train.groupby('uidx')['rating'].mean()
        self.user_mean = np.full(self.n_users, self.mu)
        self.user_mean[s.index.values] = s.values
        return self

    # ------------------------------------------------------------ 공통: (행, 이력 상품) 쌍 유사도
    def _chunks(self, uidx, iidx):
        """사용자 순으로 정렬한 뒤 (행, 그 사용자의 이력 상품) 쌍마다 코사인 유사도를 계산한다.
        - 사용자당 행이 적을 때(valid·test 예측): 쌍별 원소곱 합 (희소 행렬 그대로)
        - 사용자당 행이 많을 때(후보 100개): 사용자마다 (행 × 이력) 작은 행렬곱
        yield: (원래 행 번호, 쌍의 지역 행 번호, 유사도, 평점, 행별 이력 개수)"""
        uidx = np.asarray(uidx, dtype=np.int64)
        iidx = np.asarray(iidx, dtype=np.int64)
        order = np.argsort(uidx, kind='stable')
        us, its = uidx[order], iidx[order]
        if len(us) == 0:
            return
        nh_row = self.indptr[us + 1] - self.indptr[us]
        bounds = np.flatnonzero(np.r_[True, us[1:] != us[:-1], True])     # 사용자 경계
        starts, ends = bounds[:-1], bounds[1:]
        nh_user = nh_row[starts]
        pairs_user = (ends - starts) * nh_user
        a = 0
        while a < len(starts):
            b, tot = a, 0                                   # 쌍 수가 budget을 넘지 않게 사용자를 묶는다
            while b < len(starts) and (b == a or tot + pairs_user[b] <= self.pair_budget):
                tot += int(pairs_user[b]); b += 1
            r0, r1 = starts[a], ends[b - 1]
            uu = us[starts[a:b]]
            h_list = [np.arange(self.indptr[u], self.indptr[u + 1]) for u in uu]
            h_pos = np.concatenate(h_list) if tot > 0 else np.empty(0, dtype=np.int64)
            rows_per_user = ends[a:b] - starts[a:b]
            col_start_row = np.repeat(np.concatenate([[0], np.cumsum(nh_user[a:b])[:-1]]), rows_per_user)
            nh = nh_row[r0:r1]
            pr = np.repeat(np.arange(r1 - r0), nh)
            off = np.arange(tot) - np.repeat(np.cumsum(nh) - nh, nh)
            pc = np.repeat(col_start_row, nh) + off
            if tot == 0:
                sims = np.empty(0)
            elif (r1 - r0) <= 4 * (b - a):
                T = self.X[its[r0:r1][pr]]
                H = self.X[self.h_items[h_pos[pc]]]
                sims = np.asarray(T.multiply(H).sum(axis=1), dtype=np.float64).ravel()
            else:
                parts = []
                for j in range(b - a):
                    if len(h_list[j]) == 0:
                        continue
                    rr = its[starts[a + j]:ends[a + j]]
                    parts.append((self.X[rr] @ self.X[self.h_items[h_list[j]]].T).toarray().ravel())
                sims = np.concatenate(parts).astype(np.float64)
            rats = self.h_ratings[h_pos[pc]] if tot > 0 else np.empty(0)
            yield order[r0:r1], pr, sims, rats, nh
            a = b

    @staticmethod
    def _keep_topk(pr, sims, nh, k):
        """행마다 유사도 상위 k개만 남기는 마스크 (pr은 행 순으로 정렬되어 있음)."""
        if k is None or len(sims) == 0 or k >= nh.max():
            return np.ones(len(sims), dtype=bool)
        o = np.lexsort((-sims, pr))                       # 행 안에서 유사도 내림차순
        grp_start = np.repeat(np.cumsum(nh) - nh, nh)
        rank = np.empty(len(sims), dtype=np.int64)
        rank[o] = np.arange(len(sims)) - grp_start          # pr이 이미 정렬 → grp_start[o] == grp_start
        keep = rank < k
        return keep

    # ------------------------------------------------------------ 평점 예측
    def predict_ks(self, uidx, iidx, ks) -> dict:
        """이웃 수 k를 여러 개 한 번에 (유사도 계산은 한 번). 반환: {k: 예측 평점 배열}"""
        uidx = np.asarray(uidx, dtype=np.int64)
        out = {k: np.empty(len(uidx)) for k in ks}
        for rows, pr, sims, rats, nh in self._chunks(uidx, iidx):
            nloc = len(rows)
            fb = self.user_mean[uidx[rows]]
            for k in ks:
                keep = self._keep_topk(pr, sims, nh, k)
                num = np.bincount(pr[keep], weights=sims[keep] * rats[keep], minlength=nloc)
                den = np.bincount(pr[keep], weights=sims[keep], minlength=nloc)
                p = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), fb)
                out[k][rows] = p
        return {k: np.clip(v, RATING_MIN, RATING_MAX) for k, v in out.items()}

    def _raw(self, uidx, iidx):
        return self.predict_ks(uidx, iidx, [self.k])[self.k]

    def fallback_rate(self, uidx, iidx) -> float:
        """분모가 0이라 사용자 평균으로 대체된 비율 (점검용)."""
        uidx = np.asarray(uidx, dtype=np.int64)
        n0 = 0
        for rows, pr, sims, rats, nh in self._chunks(uidx, iidx):
            keep = self._keep_topk(pr, sims, nh, self.k)
            den = np.bincount(pr[keep], weights=sims[keep], minlength=len(rows))
            n0 += int((den <= 1e-12).sum())
        return n0 / max(len(uidx), 1)

    # ------------------------------------------------------------ Top-N 원점수
    def _agg(self, M, k):
        """M: (상품 수 × 이력 수) 유사도 → 상품별 상위 k개 평균"""
        n_h = M.shape[1]
        if n_h == 0:
            return np.zeros(M.shape[0])
        if k is None or k >= n_h:
            return M.mean(axis=1)
        if k == 1:
            return M.max(axis=1)
        return np.partition(M, n_h - k, axis=1)[:, n_h - k:].mean(axis=1)

    def score_ks(self, uidx, iidx, score_ks) -> dict:
        """score_k 여러 개 한 번에. 반환: {score_k: 원점수 배열}"""
        uidx = np.asarray(uidx, dtype=np.int64)
        out = {k: np.zeros(len(uidx)) for k in score_ks}
        for rows, pr, sims, rats, nh in self._chunks(uidx, iidx):
            nloc = len(rows)
            for k in score_ks:
                keep = self._keep_topk(pr, sims, nh, k)
                tot = np.bincount(pr[keep], weights=sims[keep], minlength=nloc)
                cnt = np.minimum(nh, nh.max() if k is None else k).astype(np.float64)
                out[k][rows] = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
        return out

    def score(self, uidx, iidx) -> np.ndarray:
        return self.score_ks(uidx, iidx, [self.score_k])[self.score_k]

    def score_rows_ks(self, uidx, score_ks) -> dict:
        """프로토콜 ③용. 반환: {score_k: (len(uidx), n_items) 원점수 행렬}"""
        uidx = np.asarray(uidx, dtype=np.int64)
        out = {k: np.empty((len(uidx), self.n_items)) for k in score_ks}
        a = 0
        while a < len(uidx):
            # 상품 수 × 이력 열 수가 rows_budget을 넘지 않게 사용자를 묶는다
            b, cols = a, 0
            while b < len(uidx):
                u = uidx[b]; c2 = cols + int(self.indptr[u + 1] - self.indptr[u])
                if b > a and c2 * self.n_items > self.rows_budget:
                    break
                cols, b = c2, b + 1
            uu = uidx[a:b]
            h_pos = [np.arange(self.indptr[u], self.indptr[u + 1]) for u in uu]
            all_pos = np.concatenate(h_pos) if cols > 0 else np.empty(0, dtype=np.int64)
            if cols > 0:
                S = (self.X @ self.X[self.h_items[all_pos]].T).toarray()     # (n_items × cols)
            c = 0
            for r, hp in enumerate(h_pos):
                M = S[:, c:c + len(hp)] if len(hp) > 0 else np.zeros((self.n_items, 0), dtype=np.float32)
                for k in score_ks:
                    out[k][a + r] = self._agg(M, k)
                c += len(hp)
            a = b
        return out

    def score_rows(self, uidx) -> np.ndarray:
        return self.score_rows_ks(uidx, [self.score_k])[self.score_k]
