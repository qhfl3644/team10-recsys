"""하이브리드 필터링 두 트랙 (B5~B7, 제안서 3.3절)

(1) 평점 예측 트랙
    r_hat = w * MF + (1 - w) * CBF          (두 예측 모두 1~5점이라 척도 변환 없음)
    w는 valid RMSE로 고른다. 사용자 구간(cold/mid/active)마다 w를 따로 고르는 변형도 둔다.
    MF는 train에 없는 상품 행을 CBF로 바꾼 fallback 버전(mf_fb)을 쓴다.

(2) Top-N 트랙
    score = sum_n a_n * z_n,   z_n = 신호 n을 사용자별로 표준화한 값 (그 사용자의 후보 안에서)
    신호: 인기도 log(1+n), CBF 유사도 원점수, MF 원점수 (clip 없음)
    가중치는 valid 프로토콜 ③(p3_users) HR@10으로 고른다 (docs/EVAL_RULES.md).

RankHybrid는 레포 공통 형식의 score(uidx, iidx)와 score_rows(uidx)를 가진다 → metrics.ranking_eval에 그대로 넣는다.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from config import RATING_MIN, RATING_MAX, SEED


# =============================================================== (1) 평점 예측 트랙
W_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def blend(p_mf, p_cbf, w):
    """w * MF + (1 - w) * CBF. w는 숫자 하나 또는 행별 배열."""
    return np.clip(np.asarray(w) * np.asarray(p_mf) + (1 - np.asarray(w)) * np.asarray(p_cbf), RATING_MIN, RATING_MAX)


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def _mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def search_w(y, p_mf, p_cbf, grid=W_GRID):
    """w 격자 탐색. 반환: (표 DataFrame[w, rmse, mae], RMSE 최소 w)"""
    t = pd.DataFrame([dict(w=float(w), rmse=_rmse(y, blend(p_mf, p_cbf, w)), mae=_mae(y, blend(p_mf, p_cbf, w))) for w in grid])
    return t, float(t.loc[t['rmse'].idxmin(), 'w'])


def search_w_by_group(y, p_mf, p_cbf, groups, grid=W_GRID):
    """구간별로 w를 따로 고른다. 반환: (표 DataFrame[group, w, rmse, mae, n], {구간: w})"""
    rows, best = [], {}
    for g in pd.unique(groups):
        m = groups == g
        t, w = search_w(y[m], p_mf[m], p_cbf[m], grid)
        t.insert(0, 'group', g); t['n'] = int(m.sum())
        rows.append(t); best[g] = w
    return pd.concat(rows, ignore_index=True), best


def blend_by_group(p_mf, p_cbf, groups, w_by_group: dict):
    w = np.array([w_by_group[g] for g in groups], dtype=np.float64)
    return blend(p_mf, p_cbf, w)


# =============================================================== (2) Top-N 트랙
def _z_rows(S):
    """(사용자 × 상품) 행렬을 행(사용자)마다 표준화. 표준편차가 0이면 0."""
    S = np.asarray(S, dtype=np.float64)
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True)
    return np.where(sd > 1e-12, (S - mu) / np.where(sd > 1e-12, sd, 1.0), 0.0)


def _z_groups(x, uidx):
    """행 단위 점수를 같은 사용자(uidx)끼리 표준화 (프로토콜 ①② 후보 100개 안에서)."""
    x = np.asarray(x, dtype=np.float64)
    _, inv, cnt = np.unique(uidx, return_inverse=True, return_counts=True)
    mu = np.bincount(inv, weights=x) / cnt
    var = np.bincount(inv, weights=x * x) / cnt - mu ** 2
    sd = np.sqrt(np.maximum(var, 0))
    return np.where(sd[inv] > 1e-12, (x - mu[inv]) / np.where(sd[inv] > 1e-12, sd[inv], 1.0), 0.0)


class RankHybrid:
    """신호 모델들의 원점수를 사용자별 z-점수로 바꿔 가중합한다.
    signals: {이름: score/score_rows를 가진 모델},  weights: {이름: 가중치}"""
    name = 'rank_hybrid'

    def __init__(self, signals: dict, weights: dict):
        self.signals = signals
        self.weights = {k: float(v) for k, v in weights.items()}
        self.n_items = next(iter(signals.values())).n_items

    def score(self, uidx, iidx) -> np.ndarray:
        uidx = np.asarray(uidx, dtype=np.int64); iidx = np.asarray(iidx, dtype=np.int64)
        out = np.zeros(len(uidx))
        for n, m in self.signals.items():
            if self.weights.get(n, 0) != 0:
                out += self.weights[n] * _z_groups(m.score(uidx, iidx), uidx)
        return out

    def score_rows(self, uidx) -> np.ndarray:
        uidx = np.asarray(uidx, dtype=np.int64)
        out = np.zeros((len(uidx), self.n_items))
        for n, m in self.signals.items():
            if self.weights.get(n, 0) != 0:
                out += self.weights[n] * _z_rows(m.score_rows(uidx))
        return out


def weight_grid(names, step=0.1):
    """합이 1인 가중치 조합 (0.1 간격, 신호 3개면 66개)."""
    n = int(round(1 / step))
    combos = [c for c in itertools.product(range(n + 1), repeat=len(names)) if sum(c) == n]
    return [dict(zip(names, np.array(c) / n)) for c in combos]


def search_rank_weights(signals: dict, data, split: str, users, grid, k: int = 10, chunk: int = 256,
                        seed: int = SEED, verbose: bool = True):
    """프로토콜 ③에서 가중치 조합마다 HR@k·NDCG@k를 계산한다.
    사용자 묶음마다 신호 z-점수를 한 번만 계산하고, 그 자리에서 모든 조합의 정답 순위를 구한다.
    정답 순위는 동점 무작위 (src/eval/ranking.py와 같은 규칙), 사용자가 평가한 다른 상품은 순위에서 뺀다.
    반환: (표 DataFrame, {조합 번호: 사용자별 hit 배열})"""
    import time
    users = np.asarray(users, dtype=np.int64)
    target = getattr(data, split).set_index('uidx')['iidx'].reindex(users).values.astype(np.int64)
    names = list(signals)
    W = np.array([[g[n] for n in names] for g in grid], dtype=np.float32)       # (조합 × 신호)
    rngs = [np.random.default_rng([seed, 3, j]) for j in range(len(grid))]
    ranks = np.empty((len(grid), len(users)), dtype=np.int64)
    t0 = time.time()
    for s0 in range(0, len(users), chunk):
        uu = users[s0:s0 + chunk]; tg = target[s0:s0 + chunk]; B = len(uu)
        Z = np.stack([_z_rows(signals[n].score_rows(uu)).astype(np.float32) for n in names])   # (신호 × B × 상품)
        er, ec = [], []
        for r, u in enumerate(uu):
            ex = data.rated_except(int(u), int(tg[r])); er.append(np.full(len(ex), r)); ec.append(ex)
        er = np.concatenate(er); ec = np.concatenate(ec)
        ar = np.arange(B)
        for j in range(len(grid)):
            S = np.tensordot(W[j], Z, axes=1)                                     # (B × 상품)
            S[er, ec] = -np.inf
            st = S[ar, tg][:, None]
            higher = (S > st).sum(axis=1)
            ties = (S == st).sum(axis=1) - 1
            ranks[j, s0:s0 + B] = 1 + higher + rngs[j].integers(0, ties + 1)
        if verbose and (s0 // chunk) % 10 == 0:
            print(f'  {min(s0 + chunk, len(users)):,}/{len(users):,}명 ({time.time()-t0:.0f}초)')
    rows, hits = [], {}
    for j, g in enumerate(grid):
        r = ranks[j]; hit = r <= k
        nd = np.where(hit, 1.0 / np.log2(r + 1), 0.0)
        rows.append({**{f'w_{n}': g[n] for n in names}, f'HR{k}': float(hit.mean()), f'NDCG{k}': float(nd.mean())})
        hits[j] = pd.Series(hit.astype(np.float64), index=users)
    return pd.DataFrame(rows), hits
