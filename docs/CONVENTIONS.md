# 팀 공통 규칙 (10/24 킥오프 합의안)

> 킥오프에서 바뀐 내용은 이 파일을 고친 뒤 커밋한다.

## 데이터 파일

| 항목 | 규칙 |
|---|---|
| 형식 | `.parquet` |
| 평점 파일 컬럼 | `user_id`(str), `item_id`(str), `rating`(float32), `timestamp`(int64, 밀리초) |
| 파일 | `ratings_all`, `train`, `valid`, `test` (데이터셋별 폴더) |
| 정수 번호 | `user_map`(user_id → uidx), `item_map`(item_id → iidx). 정렬된 ID에 0부터 번호. **팀 전체가 이 번호만 사용** |
| 상품 텍스트 | `item_features.parquet` (iidx 순서 = TF-IDF 행 순서) |
| 경로 | `config.py`의 `proc_path`, `pred_path` 사용. 경로를 코드에 직접 쓰지 않는다 |

## 모델 형식

`src/models/interface.py` 참고.

- `predict(user_id, item_ids)`: 같은 순서·길이의 예측 평점, 1~5 clip, NaN 금지
- `score(user_id, item_ids)`: Top-N용 원점수, clip 없음

## 예측값 주고받기

- 위치: `Team10_RecSys/predictions/{데이터셋}/{모델}_{split}.parquet` (예: `mf_valid.parquet`)
- 컬럼: `user_id, item_id, pred`

## 랭킹 평가 후보 (03_sampling.ipynb)

| 파일 (`processed/{ds}/candidates/`) | 컬럼 | 내용 |
|---|---|---|
| `p1_valid`, `p1_test` | uidx, iidx, label | ① 미평가 상품 무작위 99개 + 정답. 사용자당 100행, label=1이 정답 |
| `p2_valid`, `p2_test` | uidx, iidx, label | ② 정답과 같은 인기도 구간(train 평점 수 log2 구간)에서 99개 + 정답 |
| `p3_users` | uidx, user_id | ③ 전체 상품 랭킹을 할 고정 사용자 10,000명 (valid·test 공통) |
| `meta.json` | | 시드, 구간 규칙, 구간을 넓힌 사용자 비율 |

- ③에서 순위에서 빼는 상품: 사용자가 평가한 상품 중 정답을 뺀 나머지 (`sampling.protocol3_exclude`)
- 정답 순위는 `src/eval/ranking.py`의 `target_rank`(①②), `target_rank_full`(③)로 계산 → 동점은 무작위
- 상품 인기도: `processed/{ds}/item_popularity.parquet` (item_id, iidx, train_count, bucket)

## CBF·하이브리드 결과물 (07_cbf, 08_hybrid)

| 위치 | 파일 | 내용 |
|---|---|---|
| `predictions/{ds}/` | `cbf_{split}` | CBF 단독 예측 |
| | `mf_fb_{split}` | MF 예측에서 train에 없는 상품 행만 CBF로 바꾼 것 (성공 기준 ②의 MF) |
| | `hybrid_{split}`, `hybrid_grp_{split}` | 평점 하이브리드 (전체 w / 구간별 w) |
| | `*_retuned_{split}` | Toys 일반화에서 Toys valid로 다시 고른 설정의 예측 (비교용) |
| `results/cbf/` | `{ds}_cbf_config.json` 외 | k, score_k, TF-IDF 설정, 튜닝표, k 변화 그래프 |
| `results/hybrid/` | `{ds}_hybrid_config.json` 외 | w, 구간별 w, Top-N 가중치, 성공 기준 판정, 평가표 |
| | `{ds}_ranking_hits_test.parquet` | Top-N 사용자별 hit (model, protocol, uidx, hit) → 평가 담당 부트스트랩용 |

- Top-N 하이브리드는 `src/models/hybrid.py`의 `RankHybrid(신호 모델들, 가중치)`로 다시 만들 수 있다 (가중치는 `{ds}_hybrid_config.json`의 `rank_weights`)
- Toys 일반화: 07·08에서 `DS = 'Toys_and_Games'`, `TRANSFER_FROM = 'Video_Games'` → 기본 결과는 Video_Games 설정을 그대로 적용한 값

## 실험 규칙

- `SEED = 42` (numpy, random, 샘플링 모두)
- 하이퍼파라미터·가중치는 **valid로만** 선택, test는 마지막에 한 번만
- 랭킹 동점은 무작위로 순위 결정 (평가 담당 함수 사용)

## Git

- `main`에 바로 올려도 되지만, 남의 폴더 파일은 고치기 전에 말하기
- 데이터·큰 결과 파일은 커밋하지 않는다 (`.gitignore`)
- 커밋 메시지 예: `data: item_features 생성`, `cbf: k 튜닝 추가`
