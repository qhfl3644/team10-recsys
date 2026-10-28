"""공통 평가 함수 (제안서 5.1·5.2절, C2 업무). 규칙은 docs/EVAL_RULES.md.

모든 팀원이 같은 함수로 채점한다.
    1) 평점 예측 : rmse / mae, 범위(all | seen) × 사용자 구간(all | cold | mid | active)
    2) 랭킹      : 프로토콜 ①②(후보 파일) / ③(전체 상품) → HR@10, NDCG@10. 동점은 src.eval.ranking으로 무작위 처리
    3) 신뢰구간  : 사용자 단위 paired 부트스트랩 (두 모델에 같은 재표본)
    4) 판정      : 성공 기준 ①~④ (docs/EVAL_RULES.md 규칙 3)

입력은 모두 정수 번호(uidx, iidx)다. 문자열 ID 파일은 attach_index로 번호를 붙인다.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import PROC_DIR, PRED_DIR, SEED, RATING_MIN, RATING_MAX
from src.eval.ranking import target_rank, target_rank_full

K = 10
GROUPS = ['cold(3-4)', 'mid(5-9)', 'active(10+)']
SCOPES = ['all', 'seen']          # seen = 정답 상품이 train에 있는 평점만 (순수 MF 비교용)


# =============================================================== 데이터
@dataclass
class Data:
    ds: str
    n_users: int
    n_items: int
    user_map: pd.DataFrame
    item_map: pd.DataFrame
    train: pd.DataFrame            # user_id, item_id, rating, timestamp, uidx, iidx
    valid: pd.DataFrame            # uidx 순 정렬, 사용자당 1행
    test: pd.DataFrame
    train_count: np.ndarray        # 사용자별 train 평점 수 (uidx 순)
    item_count: np.ndarray         # 상품별 train 평점 수 (iidx 순)
    indptr: np.ndarray = field(repr=False)   # 사용자별 평가 상품 CSR (train+valid+test)
    indices: np.ndarray = field(repr=False)

    def group_of(self, uidx) -> np.ndarray:
        """train 평점 수 → 'cold(3-4)' / 'mid(5-9)' / 'active(10+)'"""
        c = self.train_count[np.asarray(uidx)]
        return np.where(c <= 4, GROUPS[0], np.where(c <= 9, GROUPS[1], GROUPS[2]))

    def seen_mask(self, df: pd.DataFrame) -> np.ndarray:
        """정답 상품이 train에 한 번이라도 나오는 행 (True). 순수 MF는 이 행만으로 평가한다."""
        return self.item_count[df['iidx'].values] > 0

    def rated_except(self, u: int, target: int) -> np.ndarray:
        """프로토콜 ③에서 순위에서 뺄 상품 = 사용자가 평가한 상품 중 정답 제외 (sampling.protocol3_exclude와 동일)"""
        r = self.indices[self.indptr[u]:self.indptr[u + 1]]
        return r[r != target]


def attach_index(df: pd.DataFrame, user_map: pd.DataFrame, item_map: pd.DataFrame) -> pd.DataFrame:
    """user_id·item_id → uidx·iidx 컬럼 추가. 맵에 없는 ID가 있으면 에러."""
    u = pd.Categorical(df['user_id'].astype(str), categories=user_map['user_id'].astype(str).values).codes
    i = pd.Categorical(df['item_id'].astype(str), categories=item_map['item_id'].astype(str).values).codes
    if (u < 0).any() or (i < 0).any():
        raise ValueError('user_map / item_map에 없는 ID가 있습니다')
    out = df.copy()
    out['uidx'] = u.astype(np.int32)
    out['iidx'] = i.astype(np.int32)
    return out


def load_data(ds: str, proc_dir: str = PROC_DIR) -> Data:
    p = f'{proc_dir}/{ds}'
    umap = pd.read_parquet(f'{p}/user_map.parquet')
    imap = pd.read_parquet(f'{p}/item_map.parquet')
    sp = {s: attach_index(pd.read_parquet(f'{p}/{s}.parquet'), umap, imap) for s in ['train', 'valid', 'test']}
    for s in ['valid', 'test']:
        sp[s] = sp[s].sort_values('uidx', kind='stable').reset_index(drop=True)
        if not sp[s]['uidx'].is_unique:
            raise ValueError(f'{ds} {s}: 사용자당 1행이어야 합니다')
    n_users, n_items = len(umap), len(imap)
    train_count = np.bincount(sp['train']['uidx'].values, minlength=n_users)
    item_count = np.bincount(sp['train']['iidx'].values, minlength=n_items)
    allr = pd.concat([sp[s][['uidx', 'iidx']] for s in ['train', 'valid', 'test']]).sort_values(['uidx', 'iidx'], kind='stable')
    counts = np.bincount(allr['uidx'].values, minlength=n_users)
    indptr = np.concatenate([[0], np.cumsum(counts)])
    return Data(ds, n_users, n_items, umap, imap, sp['train'], sp['valid'], sp['test'],
                train_count, item_count, indptr, allr['iidx'].values.astype(np.int32))


# =============================================================== 예측값 파일
PRED_COLUMNS = ['user_id', 'item_id', 'uidx', 'iidx', 'pred']


def save_predictions(ds: str, model: str, split: str, df: pd.DataFrame, pred: np.ndarray, pred_dir: str = PRED_DIR):
    """predictions/{ds}/{model}_{split}.parquet 저장. df는 valid 또는 test (uidx 순)."""
    if len(pred) != len(df):
        raise ValueError('pred 길이가 df와 다릅니다')
    if np.isnan(pred).any():
        raise ValueError('pred에 NaN이 있습니다')
    os.makedirs(f'{pred_dir}/{ds}', exist_ok=True)
    out = df[['user_id', 'item_id', 'uidx', 'iidx']].copy()
    out['pred'] = np.asarray(pred, dtype=np.float32)
    path = f'{pred_dir}/{ds}/{model}_{split}.parquet'
    out.to_parquet(path, index=False)
    return path


def load_predictions(ds: str, model: str, split: str, df: pd.DataFrame, pred_dir: str = PRED_DIR) -> np.ndarray:
    """저장된 예측값을 df(uidx 순)에 맞춰 돌려준다. 사용자가 빠져 있으면 에러."""
    p = pd.read_parquet(f'{pred_dir}/{ds}/{model}_{split}.parquet')
    if 'uidx' not in p.columns:
        raise ValueError('예측값 파일에 uidx 컬럼이 없습니다 (EVAL_RULES 규칙 2)')
    s = pd.Series(p['pred'].values, index=p['uidx'].values)
    out = s.reindex(df['uidx'].values).values
    if np.isnan(out).any():
        raise ValueError(f'{model}_{split}: 예측값이 없는 사용자가 {int(np.isnan(out).sum())}명')
    return out.astype(np.float64)


# =============================================================== 평점 예측
def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rating_table(preds: dict, df: pd.DataFrame, data: Data) -> pd.DataFrame:
    """preds = {모델 이름: 예측 배열(df 순서)}. 범위(all/seen) × 구간(all + 3개)별 RMSE·MAE·n."""
    y = df['rating'].values.astype(np.float64)
    seen = data.seen_mask(df)
    grp = data.group_of(df['uidx'].values)
    rows = []
    for name, p in preds.items():
        p = np.clip(np.asarray(p, dtype=np.float64), RATING_MIN, RATING_MAX)
        for scope, smask in [('all', np.ones(len(df), bool)), ('seen', seen)]:
            for g in ['all'] + GROUPS:
                m = smask & ((grp == g) if g != 'all' else True)
                rows.append(dict(model=name, scope=scope, group=g, n=int(m.sum()),
                                 rmse=rmse(y[m], p[m]), mae=mae(y[m], p[m])))
    return pd.DataFrame(rows)


# =============================================================== 랭킹
def load_candidates(ds: str, proto: str, split: str, proc_dir: str = PROC_DIR) -> pd.DataFrame:
    """p1/p2 후보 파일 (uidx, iidx, label). uidx 순 정렬해서 돌려준다."""
    c = pd.read_parquet(f'{proc_dir}/{ds}/candidates/{proto}_{split}.parquet')
    return c.sort_values('uidx', kind='stable').reset_index(drop=True)


def load_p3_users(ds: str, proc_dir: str = PROC_DIR) -> np.ndarray:
    return np.sort(pd.read_parquet(f'{proc_dir}/{ds}/candidates/p3_users.parquet')['uidx'].values.astype(np.int64))


def rank_candidates(score_fn, cand: pd.DataFrame, users: np.ndarray | None = None, seed: int = SEED):
    """프로토콜 ①②. score_fn(uidx 배열, iidx 배열) → 점수. 반환: (사용자 uidx 배열, 정답 순위 배열)"""
    if users is not None:
        cand = cand[cand['uidx'].isin(users)]
    u = cand['uidx'].values; i = cand['iidx'].values
    n_per = np.bincount(u)[np.unique(u)]
    if (n_per != n_per[0]).any():
        raise ValueError('후보 파일: 사용자마다 행 수가 같아야 합니다')
    w = int(n_per[0])
    s = np.asarray(score_fn(u, i), dtype=np.float64).reshape(-1, w)
    t = (cand['label'].values == 1).reshape(-1, w)
    return u.reshape(-1, w)[:, 0], target_rank(s, t, np.random.default_rng([seed, 1]))


def rank_full(score_rows_fn, users: np.ndarray, targets: np.ndarray, data: Data, chunk: int = 256, seed: int = SEED):
    """프로토콜 ③. score_rows_fn(uidx 배열) → (n, n_items) 점수 행렬. 사용자를 chunk명씩 끊어서 계산.
    반환: 정답 순위 배열 (users 순)"""
    rng = np.random.default_rng([seed, 3])
    out = np.empty(len(users), dtype=np.int64)
    for s in range(0, len(users), chunk):
        uu = users[s:s + chunk]
        S = np.asarray(score_rows_fn(uu), dtype=np.float64)
        if S.shape != (len(uu), data.n_items):
            raise ValueError(f'score_rows는 (n, {data.n_items}) 행렬을 돌려줘야 합니다')
        if np.isnan(S).any():
            raise ValueError('점수에 NaN이 있습니다')
        for r, u in enumerate(uu):
            t = int(targets[s + r])
            out[s + r] = target_rank_full(S[r], t, data.rated_except(int(u), t), rng)
    return out


def hr_ndcg(ranks: np.ndarray, k: int = K):
    ranks = np.asarray(ranks)
    hit = ranks <= k
    ndcg = np.where(hit, 1.0 / np.log2(ranks + 1), 0.0)
    return float(hit.mean()), float(ndcg.mean()), hit.astype(np.float64), ndcg


def ranking_eval(models: dict, data: Data, split: str, protocols=('p1', 'p2', 'p3'),
                 users: np.ndarray | None = None, proc_dir: str = PROC_DIR, chunk: int = 256):
    """models = {이름: 모델}. 모델은 score(uidx, iidx)와 (③용) score_rows(uidx)를 가진다.
    users: 평가할 사용자 (None이면 ①②는 전체, ③은 p3_users). 규칙: 프로토콜 간 비교는 같은 users로.
    반환: (표 DataFrame, {(모델, 프로토콜): 사용자별 hit 배열})"""
    target = getattr(data, split).set_index('uidx')['iidx']
    p3 = load_p3_users(data.ds, proc_dir)
    rows, hits = [], {}
    for proto in protocols:
        if proto in ('p1', 'p2'):
            cand = load_candidates(data.ds, proto, split, proc_dir)
            for name, m in models.items():
                uu, r = rank_candidates(m.score, cand, users)
                hr, nd, h, _ = hr_ndcg(r)
                rows.append(dict(model=name, protocol=proto, split=split, n_users=len(uu), HR10=hr, NDCG10=nd))
                hits[(name, proto)] = pd.Series(h, index=uu)
        elif proto == 'p3':
            uu = p3 if users is None else np.intersect1d(p3, users)
            tg = target.reindex(uu).values
            for name, m in models.items():
                r = rank_full(m.score_rows, uu, tg, data, chunk)
                hr, nd, h, _ = hr_ndcg(r)
                rows.append(dict(model=name, protocol=proto, split=split, n_users=len(uu), HR10=hr, NDCG10=nd))
                hits[(name, proto)] = pd.Series(h, index=uu)
    return pd.DataFrame(rows), hits


# =============================================================== 부트스트랩 (규칙 1)
def paired_bootstrap(a: np.ndarray, b: np.ndarray, stat: str = 'rmse', n_boot: int = 1000, seed: int = SEED):
    """같은 사용자 순서로 정렬된 두 모델의 사용자별 값(a, b)에 대해 (stat(a) − stat(b))의 95% 신뢰구간.
    stat='rmse'  : a, b = 사용자별 제곱오차 → sqrt(mean) 차이
    stat='mean'  : a, b = 사용자별 hit(0/1) 또는 NDCG → 평균 차이
    같은 재표본을 두 모델에 함께 적용한다(paired)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError('두 배열의 길이가 다릅니다 (같은 사용자로 맞추세요)')
    f = (lambda x: np.sqrt(x.mean())) if stat == 'rmse' else (lambda x: x.mean())
    rng = np.random.default_rng([seed, 7])
    n = len(a); diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[k] = f(a[idx]) - f(b[idx])
    return dict(diff=float(f(a) - f(b)), lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)), n=n)


def sq_err(y, p) -> np.ndarray:
    return (np.asarray(y, dtype=np.float64) - np.clip(np.asarray(p, dtype=np.float64), RATING_MIN, RATING_MAX)) ** 2


# =============================================================== 성공 기준 판정 (규칙 3)
def criterion_rating_vs_baselines(model_pred, baseline_preds: dict, df, data: Data, scope: str = 'all'):
    """기준 ①·④: 모델 RMSE가 기준 모델 3개(전역/사용자/상품 평균) 모두보다 낮을 것 (점추정). 차이의 CI도 보고."""
    y = df['rating'].values; m = np.ones(len(df), bool) if scope == 'all' else data.seen_mask(df)
    e_m = sq_err(y[m], np.asarray(model_pred)[m]); out = []
    for name, bp in baseline_preds.items():
        e_b = sq_err(y[m], np.asarray(bp)[m]); bs = paired_bootstrap(e_b, e_m, 'rmse')
        out.append(dict(baseline=name, scope=scope, rmse_model=float(np.sqrt(e_m.mean())), rmse_baseline=float(np.sqrt(e_b.mean())),
                        baseline_minus_model=bs['diff'], ci_lo=bs['lo'], ci_hi=bs['hi'], passed=bool(np.sqrt(e_m.mean()) < np.sqrt(e_b.mean()))))
    t = pd.DataFrame(out); t.attrs['passed'] = bool(t['passed'].all()); return t


def criterion_cold_hybrid(hybrid_pred, single_preds: dict, df, data: Data):
    """기준 ②: cold(3-4) 구간에서 하이브리드 RMSE ≤ 단일 방식 각각.
    판정: (단일 − 하이브리드)의 95% CI 상한이 0 이상이면 통과 (= 하이브리드가 유의하게 나쁘지 않음)."""
    g = data.group_of(df['uidx'].values) == GROUPS[0]; y = df['rating'].values[g]
    e_h = sq_err(y, np.asarray(hybrid_pred)[g]); out = []
    for name, sp in single_preds.items():
        e_s = sq_err(y, np.asarray(sp)[g]); bs = paired_bootstrap(e_s, e_h, 'rmse')
        out.append(dict(single=name, group=GROUPS[0], n=int(g.sum()), rmse_hybrid=float(np.sqrt(e_h.mean())), rmse_single=float(np.sqrt(e_s.mean())),
                        single_minus_hybrid=bs['diff'], ci_lo=bs['lo'], ci_hi=bs['hi'], passed=bool(bs['hi'] >= 0)))
    t = pd.DataFrame(out); t.attrs['passed'] = bool(t['passed'].all()); return t


def criterion_topn_vs_popularity(hits: dict, hybrid: str, popularity: str = 'popularity', protocol: str = 'p3'):
    """기준 ③: 프로토콜 ③에서 랭킹 하이브리드 HR@10 > 인기도.
    판정: (하이브리드 − 인기도) HR 차이의 95% CI 하한이 0보다 크면 통과 (= 유의하게 높음)."""
    h = hits[(hybrid, protocol)]; p = hits[(popularity, protocol)]
    common = h.index.intersection(p.index)
    bs = paired_bootstrap(h.reindex(common).values, p.reindex(common).values, 'mean')
    return dict(protocol=protocol, n_users=len(common), hr_hybrid=float(h.reindex(common).mean()), hr_popularity=float(p.reindex(common).mean()),
                hybrid_minus_popularity=bs['diff'], ci_lo=bs['lo'], ci_hi=bs['hi'], passed=bool(bs['lo'] > 0))


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
