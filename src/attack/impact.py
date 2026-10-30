"""공격 효과·회복 측정 (D7)

- 타깃 상품의 평균 예측 점수 변화 (prediction shift)
- 타깃 상품이 Top-10에 들어가는 사용자 비율 (HR@10)
- 탐지 → 제거 → 재학습 후 회복 정도

HR@10을 재는 사용자는 타깃을 아직 평가하지 않은 사용자만 쓴다.
이미 산 사람에게 추천하는 건 의미가 없고, 포함하면 수치가 부풀려진다.
순위 계산은 src/eval/ranking.target_rank_full을 그대로 쓴다. (동점 무작위)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import SEED
from src.eval.ranking import target_rank_full


def eligible_users(data, target: int, n_sample: int, rng) -> np.ndarray:
    """타깃 상품을 아직 평가하지 않은 사용자 중에서 n_sample명."""
    ok = [u for u in range(data.n_users)
          if target not in data.indices[data.indptr[u]:data.indptr[u + 1]]]
    ok = np.asarray(ok, dtype=np.int64)
    if len(ok) > n_sample:
        ok = np.sort(rng.choice(ok, size=n_sample, replace=False))
    return ok


def target_metrics(model, target: int, users: np.ndarray, data, k: int = 10,
                   chunk: int = 256, seed: int = SEED):
    """타깃 상품의 평균 예측 점수와 Top-k 진입 비율."""
    users = np.asarray(users, dtype=np.int64)
    rng = np.random.default_rng([seed, 13])
    scores, hits = np.empty(len(users)), np.empty(len(users), dtype=bool)
    for s in range(0, len(users), chunk):
        uu = users[s:s + chunk]
        S = np.asarray(model.score_rows(uu), dtype=np.float64)
        for r, u in enumerate(uu):
            ex = data.rated_except(int(u), target)
            scores[s + r] = S[r, target]
            hits[s + r] = target_rank_full(S[r], target, ex, rng) <= k
        del S
    return float(scores.mean()), float(hits.mean())


def recovery_row(kind: str, size: float, tlab: str, clean, attacked, recovered,
                 det: pd.Series, n_fake: int, n_removed_real: int) -> dict:
    """회복 요약표 한 줄. clean/attacked/recovered = (평균 예측 점수, HR@10)"""
    def recovery_rate(base: float, after_attack: float, after_defense: float,
                      min_effect: float) -> float:
        """1=완전 회복, 0=회복 없음, 음수=악화. 효과가 너무 작으면 비율은 NaN."""
        shift = after_attack - base
        if abs(shift) < min_effect:
            return float('nan')
        return 1.0 - abs(after_defense - base) / abs(shift)

    return dict(공격=kind, 규모=f'{size:.0%}', 가짜수=n_fake, 타깃=tlab,
                점수_정상=round(clean[0], 3), 점수_공격후=round(attacked[0], 3),
                점수_제거후=round(recovered[0], 3),
                상승폭=round(attacked[0] - clean[0], 3),
                잔존폭=round(recovered[0] - clean[0], 3),
                점수_회복률=round(recovery_rate(clean[0], attacked[0], recovered[0], 0.01), 3),
                HR_정상=round(clean[1], 3), HR_공격후=round(attacked[1], 3),
                HR_제거후=round(recovered[1], 3),
                HR_회복률=round(recovery_rate(clean[1], attacked[1], recovered[1], 0.01), 3),
                DT_F1=float(det['F1']), DT_정밀도=float(det['정밀도']),
                DT_재현율=float(det['재현율']), 오탐_정상제거=int(n_removed_real))

