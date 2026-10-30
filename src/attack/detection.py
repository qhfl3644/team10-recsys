"""가짜 프로필 탐지 (D3~D6)

사용자별 행동 피처를 뽑아 정상/가짜를 분류
    지도학습 : 의사결정나무, k-최근접 이웃
    비지도   : k-means (공격 라벨 없이 수행하는 비교군)

피처 (Burke 등이 쓰는 표준 지표)
    프로필 길이, 평균 평점, 최고점 비율, 평균 절대 편차
    RDMA      = (1/n) * sum |r - 상품평균| / 상품 평점 수
    WDMA      = (1/n) * sum |r - 상품평균| / (상품 평점 수)^2
    WDA       = sum |r - 상품평균| / 상품 평점 수          (길이로 나누지 않음)
    LengthVar = |n_u - 평균길이| / sum_k (n_k - 평균길이)^2
    DegSim    = 코사인 유사도 상위 k명 평균

DegSim 주의
    사용자 전체 쌍의 유사도는 9.5만 명에서 계산이 불가능하다. 그래서 고정된 참조 사용자 집합(ref_users)에 대해서만 유사도를 구하고 상위 k개를 평균낸다.
    모든 사용자에게 같은 기준을 적용하므로 비교에는 문제가 없다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import RATING_MAX, SEED

FEATURE_NAMES = ['프로필 길이', '평균 평점', '최고점 비율', '평균 절대 편차',
                 'RDMA', 'WDMA', 'WDA', 'LengthVar', 'DegSim']


def _rows_csr(df: pd.DataFrame, users: np.ndarray, n_items: int) -> sparse.csr_matrix:
    pos = pd.Series(np.arange(len(users)), index=users)
    d = df[df['uidx'].isin(users)]
    return sparse.csr_matrix(
        (d['rating'].to_numpy(), (pos[d['uidx'].to_numpy()].to_numpy(), d['iidx'].to_numpy())),
        shape=(len(users), n_items))


def _l2norm(X: sparse.csr_matrix) -> sparse.csr_matrix:
    n = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel()
    n[n == 0] = 1.0
    return sparse.diags(1.0 / n) @ X


def degsim(df: pd.DataFrame, users: np.ndarray, ref_users: np.ndarray, n_items: int,
           k: int = 20, chunk: int = 2000) -> np.ndarray:
    """users 각각에 대해 ref_users와의 코사인 유사도 상위 k개 평균."""
    Xu = _l2norm(_rows_csr(df, users, n_items))
    Xr = _l2norm(_rows_csr(df, ref_users, n_items))
    ref_pos = pd.Series(np.arange(len(ref_users)), index=ref_users)
    k = max(min(k, len(ref_users) - 1), 1)

    out = np.zeros(len(users))
    for a in range(0, len(users), chunk):
        b = min(a + chunk, len(users))
        S = (Xu[a:b] @ Xr.T).toarray()
        # 자기 자신이 참조 집합에 있으면 제외
        here = ref_pos.reindex(users[a:b]).to_numpy()
        hit = ~np.isnan(here)
        if hit.any():
            S[np.where(hit)[0], here[hit].astype(int)] = -1.0
        out[a:b] = np.partition(S, -k, axis=1)[:, -k:].mean(axis=1)
        del S
    return out


def features(df: pd.DataFrame, users: np.ndarray, n_items: int,
             item_mean: np.ndarray, item_count: np.ndarray,
             ref_users: np.ndarray, degsim_k: int = 20):
    """users 순서대로 (len(users), 9) 피처 행렬."""
    users = np.asarray(users, dtype=np.int64)
    pos = pd.Series(np.arange(len(users)), index=users)
    d = df[df['uidx'].isin(users)]
    row = pos[d['uidx'].to_numpy()].to_numpy()
    col = d['iidx'].to_numpy()
    r = d['rating'].to_numpy(dtype=np.float64)

    dev = np.abs(r - item_mean[col])
    cnt = np.maximum(item_count[col], 1).astype(np.float64)

    n = len(users)
    acc = np.zeros((n, 6))
    np.add.at(acc[:, 0], row, 1.0)                                  # 길이
    np.add.at(acc[:, 1], row, r)                                    # 평점 합
    np.add.at(acc[:, 2], row, (r >= RATING_MAX).astype(float))      # 최고점 개수
    np.add.at(acc[:, 3], row, dev)                                  # 편차 합
    np.add.at(acc[:, 4], row, dev / cnt)                            # RDMA/WDA 분자
    np.add.at(acc[:, 5], row, dev / cnt ** 2)                       # WDMA 분자

    length = np.maximum(acc[:, 0], 1.0)
    mean_len = length.mean()
    denom = max(float(((length - mean_len) ** 2).sum()), 1e-9)

    F = np.column_stack([
        acc[:, 0],
        acc[:, 1] / length,
        acc[:, 2] / length,
        acc[:, 3] / length,
        acc[:, 4] / length,                       # RDMA
        acc[:, 5] / length,                       # WDMA
        acc[:, 4],                                # WDA
        np.abs(length - mean_len) / denom,        # LengthVar
        degsim(df, users, ref_users, n_items, k=degsim_k),
    ])
    return F, list(FEATURE_NAMES)


def detect(F: np.ndarray, y: np.ndarray, names=None, n_splits: int = 5, seed: int = SEED):
    """교차검증 held-out 예측으로 세 탐지기를 비교한다 (성공 기준 ⑤).

    F·y는 평가 대상 사용자 전체여야 한다. 가짜만 전부 넣고 정상은 일부만 넣으면 양성 비율이 실제 공격 규모와 달라져 F1을 그대로 쓸 수 없다.

    반환: (비교표, 피처 중요도, 의사결정나무 held-out 예측)
    """
    names = list(names or FEATURE_NAMES)
    y = np.asarray(y, dtype=int)
    class_counts = np.bincount(y, minlength=2)
    min_count = int(class_counts.min())
    if min_count < 2:
        raise ValueError(
            f'교차검증에는 정상·가짜 사용자가 각각 2명 이상 필요합니다 '
            f'(정상={class_counts[0]}, 가짜={class_counts[1]}). '
            'N_EVAL을 늘리거나 공격 규모를 확인하세요.'
        )
    n_splits = min(n_splits, min_count)
    pred = {k: np.zeros(len(y), dtype=int) for k in ['Decision Tree', 'kNN', 'k-means']}
    importances = np.zeros(F.shape[1])

    for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(F, y):
        sc = StandardScaler().fit(F[tr])
        Xtr, Xte = sc.transform(F[tr]), sc.transform(F[te])

        dt = DecisionTreeClassifier(max_depth=6, class_weight='balanced',
                                    random_state=seed).fit(F[tr], y[tr])
        pred['Decision Tree'][te] = dt.predict(F[te])
        importances += dt.feature_importances_

        pred['kNN'][te] = KNeighborsClassifier(n_neighbors=5).fit(Xtr, y[tr]).predict(Xte)

        km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xtr)
        lab = km.predict(Xte)
        m1 = y[tr][km.labels_ == 1].mean() if (km.labels_ == 1).any() else 0.0
        m0 = y[tr][km.labels_ == 0].mean() if (km.labels_ == 0).any() else 0.0
        pred['k-means'][te] = lab if m1 >= m0 else 1 - lab

    rows = []
    for name, p in pred.items():
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, average='binary', zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        rows.append(dict(탐지기=name, 정밀도=round(pr, 3), 재현율=round(rc, 3), F1=round(f1, 3),
                         TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn)))
    imp = sorted(zip(names, importances / n_splits), key=lambda x: -x[1])
    return pd.DataFrame(rows), imp, pred['Decision Tree']


def random_f1_bound(pos_rate: float) -> float:
    """양성 비율 p에서 무작위 분류기가 낼 수 있는 F1 상한 (재현율 1일 때)"""
    return 2 * pos_rate / (pos_rate + 1)

