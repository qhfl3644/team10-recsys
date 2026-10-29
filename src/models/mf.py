"""행렬 분해 (B1~B3)

목적함수 = 제곱오차 + L2 정규화, 확률적 경사하강법(SGD)으로 최적화한다.
    r_hat(u,i) = mu + b_u + b_i + P[u] . Q[i]

- k=0으로 두면 잠재요인 없이 편향만 쓰는 모델이 된다 (구성요소 분석용)
- reg_bi를 따로 주면 상품 편향만 더 강하게 수축시킬 수 있다 
- 조기 종료는 valid RMSE 기준

레포 공통 형식(docs/EVAL_RULES.md 규칙 2)을 따른다.
    predict(uidx, iidx)  1~5 clip
    score(uidx, iidx)    clip 없음
    score_rows(uidx)     (len(uidx), n_items)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import SEED
from src.models.baselines import _Base


class MF(_Base):
    name = 'mf'

    def __init__(self, k: int = 16, lr: float = 0.01, reg: float = 0.1,
                 reg_bi: float | None = None, epochs: int = 60,
                 patience: int = 5, seed: int = SEED, verbose: bool = True):
        self.k, self.lr, self.reg = int(k), float(lr), float(reg)
        self.reg_bi = self.reg if reg_bi is None else float(reg_bi)
        self.epochs, self.patience = int(epochs), int(patience)
        self.seed, self.verbose = int(seed), bool(verbose)
        self.history: list[float] = []
        self.best_epoch = 0

    # ------------------------------------------------------------ 학습
    def fit(self, train: pd.DataFrame, n_users: int, n_items: int, valid: pd.DataFrame | None = None):
        for c in ['uidx', 'iidx', 'rating']:
            if c not in train.columns:
                raise ValueError(f'train에 {c} 컬럼이 필요합니다 (metrics.attach_index 사용)')
        self.n_users, self.n_items = int(n_users), int(n_items)

        rs = np.random.default_rng(self.seed)
        kk = max(self.k, 1)
        self.P = rs.normal(0, 0.05, (self.n_users, kk))
        self.Q = rs.normal(0, 0.05, (self.n_items, kk))
        self.bu = np.zeros(self.n_users)
        self.bi = np.zeros(self.n_items)

        u = train['uidx'].values.astype(np.int64)
        i = train['iidx'].values.astype(np.int64)
        r = train['rating'].values.astype(np.float64)
        self.mu = float(r.mean())

        yv = pv = None
        if valid is not None:
            yv = valid['rating'].values.astype(np.float64)
            uv = valid['uidx'].values.astype(np.int64)
            iv = valid['iidx'].values.astype(np.int64)

        order = np.arange(len(r))
        shuf = np.random.default_rng([self.seed, 11])
        best, bad, state = np.inf, 0, None

        for ep in range(1, self.epochs + 1):
            shuf.shuffle(order)
            self._epoch(u[order], i[order], r[order])
            if valid is None:
                continue
            pv = np.clip(self._raw(uv, iv), 1.0, 5.0)
            err = float(np.sqrt(np.mean((yv - pv) ** 2)))
            self.history.append(err)
            if self.verbose and (ep <= 3 or ep % 5 == 0):
                print(f'    epoch {ep:3d}  valid RMSE {err:.4f}')
            if err < best - 1e-5:
                best, bad, self.best_epoch = err, 0, ep
                state = (self.P.copy(), self.Q.copy(), self.bu.copy(), self.bi.copy())
            else:
                bad += 1
                if bad >= self.patience:
                    if self.verbose:
                        print(f'    조기 종료: {ep}에폭 (최저 {self.best_epoch}에폭, valid RMSE {best:.4f})')
                    break
        if state is not None:
            self.P, self.Q, self.bu, self.bi = state

        # train에 한 번도 안 나온 상품은 잠재요인이 초기 난수 그대로라 예측에 잡음이 섞인다.
        # 0으로 두면 그 상품의 예측이 정확히 mu + b_u 가 된다 (편향도 0이므로).
        seen = np.bincount(i, minlength=self.n_items) > 0
        self.Q[~seen] = 0.0
        self.unseen_items = int((~seen).sum())

        self.best_valid = float(best)
        return self

    def _epoch(self, u, i, r):
        lr, reg, rbi, use_f = self.lr, self.reg, self.reg_bi, self.k > 0
        P, Q, bu, bi, mu = self.P, self.Q, self.bu, self.bi, self.mu
        for n in range(len(r)):
            uu, ii = u[n], i[n]
            pred = mu + bu[uu] + bi[ii]
            if use_f:
                pred += P[uu] @ Q[ii]
            e = r[n] - pred
            bu[uu] += lr * (e - reg * bu[uu])
            bi[ii] += lr * (e - rbi * bi[ii])
            if use_f:
                pu = P[uu].copy()
                P[uu] += lr * (e * Q[ii] - reg * pu)
                Q[ii] += lr * (e * pu - reg * Q[ii])

    # ------------------------------------------------------------ 공통 형식
    def _raw(self, uidx, iidx):
        u = np.asarray(uidx, dtype=np.int64)
        i = np.asarray(iidx, dtype=np.int64)
        out = self.mu + self.bu[u] + self.bi[i]
        if self.k > 0:
            out = out + np.einsum('ij,ij->i', self.P[u], self.Q[i])
        return out

    def score_rows(self, uidx) -> np.ndarray:
        """(len(uidx), n_items) 점수 행렬. 프로토콜 ③용, clip 없음."""
        u = np.asarray(uidx, dtype=np.int64)
        out = self.mu + self.bu[u][:, None] + self.bi[None, :]
        if self.k > 0:
            out = out + self.P[u] @ self.Q.T
        return out
