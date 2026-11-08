"""두 데이터셋의 사용자·상품 중복 (보고서 2.4절, A 업무).

구축용(Video_Games)과 테스트용(Toys_and_Games)에 같은 사용자·상품이 얼마나 함께 나오는지 센다.
Amazon Reviews 2023의 user_id와 parent_asin(processed에서는 item_id)은 원본 ID 그대로이므로
문자열 ID를 직접 비교한다. 팀 정수 번호(uidx·iidx)는 데이터셋마다 따로 매긴 번호라 비교에 쓰지 않는다.

입력은 processed/{ds}/의 평점 파일(ratings_all, train)이다. 실행 예: notebooks/02b_dataset_overlap.ipynb
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import GROUPS   # ['cold(3-4)', 'mid(5-9)', 'active(10+)']


# ---------------------------------------------------------------- ID 집합
def unique_ids(df: pd.DataFrame, col: str) -> pd.Index:
    """col 열의 고유 ID (문자열, 정렬)."""
    return pd.Index(pd.unique(df[col].astype(str))).sort_values()


def common_ids(ratings: dict[str, pd.DataFrame], col: str) -> pd.Index:
    """두 데이터셋에 모두 나오는 ID (정렬). ratings는 {데이터셋 이름: 평점 DataFrame} 두 개."""
    a, b = _two(ratings)
    return unique_ids(ratings[a], col).intersection(unique_ids(ratings[b], col)).sort_values()


def overlap_table(ratings: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """사용자·상품별 중복 요약.

    열: 대상 | {a} | {b} | 공통 | {a} 대비(%) | {b} 대비(%) | {a} 평점 중 공통 몫(%) | {b} 평점 중 공통 몫(%)
    - '대비(%)'는 공통 ID 수 / 그 데이터셋의 고유 ID 수
    - '평점 중 공통 몫(%)'은 공통 ID가 들어간 평점 행 / 그 데이터셋의 전체 평점 행
    """
    a, b = _two(ratings)
    rows = []
    for col, label in [('user_id', '사용자'), ('item_id', '상품')]:
        ia, ib = unique_ids(ratings[a], col), unique_ids(ratings[b], col)
        common = ia.intersection(ib)
        rows.append({
            '대상': label, a: len(ia), b: len(ib), '공통': len(common),
            f'{a} 대비(%)': 100 * len(common) / len(ia),
            f'{b} 대비(%)': 100 * len(common) / len(ib),
            f'{a} 평점 중 공통 몫(%)': 100 * ratings[a][col].astype(str).isin(common).mean(),
            f'{b} 평점 중 공통 몫(%)': 100 * ratings[b][col].astype(str).isin(common).mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 사용자 구간
def group_of_count(train_count) -> np.ndarray:
    """train 평점 수 → 구간 이름. src.eval.metrics.Data.group_of와 같은 기준 (3~4 / 5~9 / 10+)."""
    c = np.asarray(train_count)
    return np.where(c <= 4, GROUPS[0], np.where(c <= 9, GROUPS[1], GROUPS[2]))


def user_groups(train: pd.DataFrame) -> pd.Series:
    """user_id(문자열) → 구간 이름. train에 나온 사용자 전부."""
    cnt = train['user_id'].astype(str).value_counts()
    return pd.Series(group_of_count(cnt.to_numpy()), index=cnt.index, name='group')


def group_crosstab(train_a: pd.DataFrame, train_b: pd.DataFrame, common_users: pd.Index,
                   names: tuple[str, str]) -> pd.DataFrame:
    """공통 사용자가 두 데이터셋에서 각각 어느 구간에 속하는지 교차표 (합계 행·열 포함).

    행 = names[0]의 구간, 열 = names[1]의 구간. 5-core + leave-last-out이라 모든 사용자는 train 평점이 3개 이상이다.
    """
    ga = user_groups(train_a).reindex(common_users)
    gb = user_groups(train_b).reindex(common_users)
    if ga.isna().any() or gb.isna().any():
        raise ValueError('공통 사용자 중 train 평점이 없는 사용자가 있습니다. 평점 파일과 train 파일이 같은 버전인지 확인하세요')
    ct = pd.DataFrame(0, index=GROUPS, columns=GROUPS, dtype=np.int64)
    if len(common_users):
        vc = pd.DataFrame({'a': ga.to_numpy(), 'b': gb.to_numpy()}).value_counts()
        for (ra, rb), n in vc.items():
            ct.loc[ra, rb] = int(n)
    ct['합계'] = ct.sum(axis=1)
    ct.loc['합계'] = ct.sum(axis=0)
    ct.index.name = f'{names[0]} 구간'
    ct.columns.name = f'{names[1]} 구간'
    return ct


def share_by_group(train: pd.DataFrame, common_users: pd.Index) -> pd.DataFrame:
    """한 데이터셋의 사용자 중 공통 사용자 비율을 전체·구간별로.

    leave-last-out이라 test 평점 1개 = 사용자 1명이므로, 이 비율은 test 평점 중 공통 사용자 몫과 같다.
    """
    g = user_groups(train)
    is_c = g.index.isin(common_users)
    rows = []
    for name in ['all'] + GROUPS:
        m = np.ones(len(g), bool) if name == 'all' else (g.to_numpy() == name)
        n, k = int(m.sum()), int((m & is_c).sum())
        rows.append({'group': name, '사용자 수': n, '공통 사용자 수': k, '공통 비율(%)': 100 * k / n if n else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 공통 사용자의 특징
def common_vs_rest(ratings: dict[str, pd.DataFrame], common_users: pd.Index) -> pd.DataFrame:
    """데이터셋마다 공통 사용자와 나머지 사용자의 활동량·평점 비교 (ratings_all 기준)."""
    rows = []
    for ds, df in ratings.items():
        uid = df['user_id'].astype(str)
        is_c = uid.isin(common_users).to_numpy()
        for label, m in [('공통', is_c), ('나머지', ~is_c)]:
            per_user = uid[m].value_counts()
            r = df['rating'].to_numpy()[m]
            rows.append({
                'dataset': ds, '사용자': label, '사용자 수': int(per_user.size), '평점 수': int(m.sum()),
                '사용자당 평점 평균': float(per_user.mean()) if per_user.size else np.nan,
                '사용자당 평점 중앙값': float(per_user.median()) if per_user.size else np.nan,
                '평균 평점': float(r.mean()) if r.size else np.nan,
                '5점 비율(%)': float(100 * (r == 5).mean()) if r.size else np.nan,
            })
    return pd.DataFrame(rows)


def user_mean_corr(ratings_a: pd.DataFrame, ratings_b: pd.DataFrame, common_users: pd.Index) -> dict:
    """공통 사용자의 사용자별 평균 평점이 두 데이터셋에서 얼마나 비슷한지 (Pearson, Spearman).

    사용자 편향(후하게 주는 사람·짜게 주는 사람)이 카테고리를 넘어 유지되는지 보는 참고 지표다.
    """
    def per_user_mean(df):
        return df.groupby(df['user_id'].astype(str))['rating'].mean().reindex(common_users)
    ma, mb = per_user_mean(ratings_a), per_user_mean(ratings_b)
    ok = len(common_users) >= 2
    return {
        'n_users': int(len(common_users)),
        'mean_rating_a': float(ma.mean()) if len(common_users) else np.nan,
        'mean_rating_b': float(mb.mean()) if len(common_users) else np.nan,
        'pearson': float(ma.corr(mb)) if ok else np.nan,
        'spearman': float(ma.corr(mb, method='spearman')) if ok else np.nan,
    }


def _two(ratings: dict) -> tuple[str, str]:
    if len(ratings) != 2:
        raise ValueError(f'데이터셋 두 개가 필요합니다: {list(ratings)}')
    a, b = list(ratings)
    return a, b
