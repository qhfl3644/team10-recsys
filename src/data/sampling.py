"""랭킹 평가용 샘플링 프로토콜 ①②③ (제안서 2.1절, A 업무).

- ① 사용자가 train·valid·test 어디에서도 평가하지 않은 상품 중 무작위 99개
- ② 정답과 같은 인기도 구간(train 평점 수 기준)에서 99개
- ③ 후보를 뽑지 않음. 고정한 사용자 일부에 대해 전체 상품 랭킹

모든 함수는 정수 번호(uidx, iidx)로 동작한다. 번호는 processed/{ds}/user_map, item_map 기준.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_NEG = 99


# ---------------------------------------------------------------- 데이터 읽기
def load_maps(proc_dir: str, ds: str):
    umap = pd.read_parquet(f'{proc_dir}/{ds}/user_map.parquet')
    imap = pd.read_parquet(f'{proc_dir}/{ds}/item_map.parquet')
    return umap, imap


def to_index(df: pd.DataFrame, umap: pd.DataFrame, imap: pd.DataFrame) -> pd.DataFrame:
    """user_id·item_id 문자열을 uidx·iidx 정수로 바꾼다. 맵에 없는 ID가 있으면 에러."""
    u = pd.Categorical(df['user_id'].astype(str), categories=umap['user_id'].astype(str).values).codes
    i = pd.Categorical(df['item_id'].astype(str), categories=imap['item_id'].astype(str).values).codes
    if (u < 0).any() or (i < 0).any():
        raise ValueError('user_map / item_map에 없는 ID가 있습니다')
    return pd.DataFrame({'uidx': u.astype(np.int32), 'iidx': i.astype(np.int32)})


def load_indexed_splits(proc_dir: str, ds: str):
    """train / valid / test를 정수 번호로 읽는다. valid·test는 uidx 순으로 정렬."""
    umap, imap = load_maps(proc_dir, ds)
    out = {}
    for s in ['train', 'valid', 'test']:
        df = pd.read_parquet(f'{proc_dir}/{ds}/{s}.parquet', columns=['user_id', 'item_id'])
        idx = to_index(df, umap, imap)
        out[s] = idx.sort_values('uidx', kind='stable').reset_index(drop=True)
    return out, len(umap), len(imap)


def user_item_csr(frames: list[pd.DataFrame], n_users: int):
    """여러 분할을 합쳐 사용자별 평가 상품 목록을 CSR(indptr, indices) 형태로 만든다.
    indices는 사용자 안에서 정렬되어 있어 np.isin / searchsorted가 빠르다."""
    df = pd.concat(frames, ignore_index=True).sort_values(['uidx', 'iidx'], kind='stable')
    counts = np.bincount(df['uidx'].values, minlength=n_users)
    indptr = np.concatenate([[0], np.cumsum(counts)])
    return indptr, df['iidx'].values.astype(np.int32)


def rated_of(indptr, indices, u: int) -> np.ndarray:
    return indices[indptr[u]:indptr[u + 1]]


# ---------------------------------------------------------------- 인기도
def item_popularity(train: pd.DataFrame, n_items: int) -> np.ndarray:
    """상품별 train 평점 수 (iidx 순서)."""
    return np.bincount(train['iidx'].values, minlength=n_items).astype(np.int32)


def popularity_bucket(counts: np.ndarray) -> np.ndarray:
    """log2 구간: 0개 → 0, 1개 → 1, 2~3개 → 2, 4~7개 → 3, 8~15개 → 4, ..."""
    b = np.zeros_like(counts, dtype=np.int16)
    pos = counts > 0
    b[pos] = np.floor(np.log2(counts[pos])).astype(np.int16) + 1
    return b


# ---------------------------------------------------------------- 공통: 거절 샘플링
def _draw(rng, pool_size: int, exclude_pos: np.ndarray, k: int, pool=None) -> np.ndarray:
    """0..pool_size-1(또는 pool 배열의 위치)에서 exclude_pos를 빼고 중복 없이 k개.
    exclude_pos는 정렬된 '풀 안의 위치' 배열. 뽑힌 순서 자체가 무작위다."""
    out = np.empty(0, dtype=np.int64)
    need = k
    while need > 0:
        c = rng.integers(0, pool_size, size=need + len(exclude_pos) + 16)
        c = c[~np.isin(c, exclude_pos, assume_unique=False)]
        c = np.concatenate([out, c])
        _, first = np.unique(c, return_index=True)      # 중복 제거 (등장 순서 유지)
        out = c[np.sort(first)][:k]
        need = k - len(out)
    return out if pool is None else pool[out]


def _to_long(users, targets, negs, rng) -> pd.DataFrame:
    """(사용자, 정답, 오답 99개) → 사용자당 100행. 정답 위치가 순서로 드러나지 않게 행을 섞는다."""
    n = len(users)
    items = np.concatenate([targets[:, None], negs], axis=1)            # (n, 100)
    labels = np.zeros_like(items, dtype=np.int8); labels[:, 0] = 1
    perm = np.argsort(rng.random(items.shape), axis=1)
    items = np.take_along_axis(items, perm, axis=1)
    labels = np.take_along_axis(labels, perm, axis=1)
    return pd.DataFrame({
        'uidx': np.repeat(users, items.shape[1]).astype(np.int32),
        'iidx': items.ravel().astype(np.int32),
        'label': labels.ravel(),
    })


# ---------------------------------------------------------------- 프로토콜 ①
def sample_protocol1(target_df: pd.DataFrame, indptr, indices, n_items: int, rng) -> pd.DataFrame:
    """target_df: (uidx, iidx) 사용자당 1행 (valid 또는 test 정답).
    오답 후보 = 사용자가 train·valid·test에서 평가한 상품을 모두 뺀 나머지."""
    users = target_df['uidx'].values
    targets = target_df['iidx'].values
    negs = np.empty((len(users), N_NEG), dtype=np.int32)
    for r, u in enumerate(users):
        negs[r] = _draw(rng, n_items, rated_of(indptr, indices, u), N_NEG)
    return _to_long(users, targets, negs, rng)


# ---------------------------------------------------------------- 프로토콜 ②
def sample_protocol2(target_df: pd.DataFrame, indptr, indices, bucket: np.ndarray, rng,
                     min_pool: int = N_NEG):
    """정답과 같은 인기도 구간에서 오답 99개. 그 구간에 뽑을 상품이 부족하면
    이웃 구간을 한 칸씩 넓힌다. 반환: (long DataFrame, 사용자별 [구간 하한, 상한])"""
    order = np.argsort(bucket, kind='stable')               # 구간 순으로 정렬한 상품 번호
    sorted_b = bucket[order]
    n_b = int(bucket.max()) + 1
    start = np.searchsorted(sorted_b, np.arange(n_b), side='left')
    end = np.searchsorted(sorted_b, np.arange(n_b), side='right')
    pos_of_item = np.empty_like(order); pos_of_item[order] = np.arange(len(order))

    users = target_df['uidx'].values
    targets = target_df['iidx'].values
    negs = np.empty((len(users), N_NEG), dtype=np.int32)
    ranges = np.empty((len(users), 2), dtype=np.int16)
    for r, (u, t) in enumerate(zip(users, targets)):
        rated = rated_of(indptr, indices, u)
        lo = hi = int(bucket[t])
        while True:
            s, e = start[lo], end[hi]
            rp = pos_of_item[rated]
            rp = np.sort(rp[(rp >= s) & (rp < e)]) - s       # 풀 안에 있는 평가 상품 위치
            if (e - s) - len(rp) >= min_pool or (lo == 0 and hi == n_b - 1):
                break
            if lo > 0: lo -= 1
            if hi < n_b - 1: hi += 1
        negs[r] = _draw(rng, e - s, rp, N_NEG, pool=order[s:e])
        ranges[r] = (lo, hi)
    return _to_long(users, targets, negs, rng), ranges


# ---------------------------------------------------------------- 프로토콜 ③
def select_protocol3_users(n_users: int, n_sample: int, rng) -> np.ndarray:
    """전체 랭킹을 할 사용자를 고정해서 뽑는다 (valid·test 같은 사용자)."""
    n = min(n_sample, n_users)
    return np.sort(rng.choice(n_users, size=n, replace=False)).astype(np.int32)


def protocol3_exclude(indptr, indices, u: int, target: int) -> np.ndarray:
    """③에서 순위에서 뺄 상품 = 사용자가 평가한 상품 중 정답을 뺀 나머지
    (①과 같은 기준: 다른 분할에서 평가한 상품은 후보가 아님)."""
    rated = rated_of(indptr, indices, u)
    return rated[rated != target]
