# 평가 규칙 (C 업무, 10/27 제안 → 단톡 합의 후 확정)

> 제안서(영문 제출본) 1.4·2.1·5.1·5.2절을 코드로 옮길 때 정해야 했던 세 가지와, 지표 함수 사용법.
> 코드: `src/eval/metrics.py`, `src/models/baselines.py`, 실행 예: `notebooks/04_baselines_eval.ipynb`

## 규칙 1. 신뢰구간은 평가 담당이 한꺼번에, 사용자 단위 paired 부트스트랩으로 계산한다

- 성공 기준 ②·③은 두 모델의 **차이**(단일 − 하이브리드, 하이브리드 − 인기도)에 대한 신뢰구간을 요구한다.
  차이의 신뢰구간은 **같은 사용자 재표본을 두 모델에 함께 적용**해야 나온다(paired). 각자 자기 모델만 따로 돌리면 재표본이 달라 차이의 구간을 만들 수 없다.
- 그래서 모델 담당은 **예측값 파일만** 올리고(규칙 2), 평가 담당이 `metrics.paired_bootstrap`으로 모든 모델을 한 번에 계산한다.
- 설정: 사용자 단위 재표본(leave-last-out이라 테스트 평점 1개 = 사용자 1명), B = 1,000, 95% 백분위 구간, 시드 `SEED`.
- 랭킹 지표의 구간도 같은 방식(사용자별 hit 0/1의 평균 차이).

## 규칙 2. 점수는 정수 번호 배열로 한꺼번에 주고받는다

- 모델 메서드 (모든 모델 공통, `src/models/baselines.py` 참고)
  - `predict(uidx, iidx)` → 같은 길이의 예측 평점, 1~5 clip, NaN 없음 (평점 트랙)
  - `score(uidx, iidx)` → Top-N 원점수, clip 없음 (프로토콜 ①②)
  - `score_rows(uidx)` → `(len(uidx), n_items)` 행렬 (프로토콜 ③ 전체 상품 랭킹). MF는 `P[uidx] @ Q.T + 편향`.
- `interface.py`의 문자열 ID·한 명씩 형식은 그대로 두되, 필요하면 `baselines.IdAdapter`로 감싼다. 평가 코드는 정수 번호만 쓴다. (사용자 한 명씩 부르면 ①②만으로도 Video_Games 9.5만 번, Toys 43만 번 호출이라 느리다.)
- 예측값 파일 `predictions/{ds}/{모델}_{split}.parquet` 컬럼: **`user_id, item_id, uidx, iidx, pred`** (기존 3개에 번호 2개 추가). `metrics.save_predictions`를 쓰면 형식이 맞는다. 행 순서는 valid/test 파일의 uidx 순.
- Top-N 트랙 결과는 예측값이 아니라 **모델 객체**(또는 `score`/`score_rows`를 가진 래퍼)로 넘긴다. ③은 사용자당 상품 전체 점수가 필요해서 파일로 주고받기 어렵다.

## 규칙 3. 성공 기준 판정 규칙 (결과를 보기 전에 고정)

| 기준 | 판정 | 함수 |
|---|---|---|
| ① 구축용 RMSE < 기준 모델 3개 | 점추정 RMSE가 전역·사용자·상품 평균 **모두보다 낮으면** 통과. 차이의 95% CI를 같이 보고 | `criterion_rating_vs_baselines` |
| ② cold(3~4) 구간 하이브리드 RMSE ≤ 단일 방식 각각 | **(단일 − 하이브리드)의 95% CI 상한 ≥ 0이면 통과** (= 하이브리드가 유의하게 나쁘지 않음). 두 데이터셋 모두, MF는 fallback 버전 | `criterion_cold_hybrid` |
| ③ 프로토콜 ③ HR@10: 랭킹 하이브리드 > 인기도 | **(하이브리드 − 인기도)의 95% CI 하한 > 0이면 통과** (= 유의하게 높음). `p3_users` 공통 표본 | `criterion_topn_vs_popularity` |
| ④ 테스트셋, 구축용 하이퍼파라미터로 RMSE < 기준 모델 3개 | ①과 같은 규칙을 Toys_and_Games에 적용 | `criterion_rating_vs_baselines` |
| ⑤ 공격 3% 이상 DT F1 ≥ 0.7 | 교차검증 held-out 예측의 F1 (혜림님 D 코드) | — |
| ⑥ 회복 정도 | 판정 없음. 타깃 인기도 × 탐지 재현율별로 표로 보고 | — |

②는 "낮거나 같을 것"이라 비겨도 통과, ③은 "높을 것"이라 이겨야 통과로 둔 것이다. 지난 사전 실험에서 Toys cold 구간이 MF 1.1729 vs 하이브리드 1.1731(CI에 0 포함)처럼 나온 적이 있어, 이런 경우를 "통과"로 볼지 미리 정해 둔다.

## 평가 범위·구간 (제안서 2.1절 반영)

- **scope = all | seen.** `seen`은 정답 상품이 train에 있는 테스트 평점만(순수 MF 비교용). 순수 MF를 보고할 때는 기준 모델·CBF도 `seen`으로 다시 계산한다. `rating_table`이 둘 다 낸다.
- **group = cold(3-4) | mid(5-9) | active(10+)**: train 평점 수 기준 (`Data.group_of`).
- **랭킹 사용자 표본:** ③은 `p3_users` 1만 명. 프로토콜 간 비교와 기준 ③ 판정은 **①②도 같은 1만 명으로 잘라서** 계산한다(`ranking_eval(users=p3_users)`). 전체 사용자 ①② 결과는 참고용으로 같이 낸다.
- **동점:** `src/eval/ranking.py`(무작위) 그대로. 평가 시드는 `[SEED, 1]`(①②), `[SEED, 3]`(③), `[SEED, 7]`(부트스트랩).

## 사용 예

```python
from src.eval import metrics as M
from src.models.baselines import BASELINES

data = M.load_data('Video_Games')                       # train/valid/test + uidx·iidx, 구간, CSR
models = {k: c().fit(data.train, data.n_users, data.n_items) for k, c in BASELINES.items()}

# 평점 트랙: 예측값 파일로
preds = {k: M.load_predictions('Video_Games', k, 'test', data.test) for k in ['global_mean', 'user_mean', 'item_mean', 'mf', 'cbf', 'hybrid']}
M.rating_table(preds, data.test, data)                   # scope × group별 RMSE·MAE
M.criterion_cold_hybrid(preds['hybrid'], {'mf': preds['mf'], 'cbf': preds['cbf']}, data.test, data)

# Top-N 트랙: 모델 객체로
p3 = M.load_p3_users('Video_Games')
table, hits = M.ranking_eval({**models, 'rank_hybrid': rank_hybrid_model}, data, 'test', users=p3)
M.criterion_topn_vs_popularity(hits, hybrid='rank_hybrid')
```
