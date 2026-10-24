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

## 실험 규칙

- `SEED = 42` (numpy, random, 샘플링 모두)
- 하이퍼파라미터·가중치는 **valid로만** 선택, test는 마지막에 한 번만
- 랭킹 동점은 무작위로 순위 결정 (평가 담당 함수 사용)

## Git

- `main`에 바로 올려도 되지만, 남의 폴더 파일은 고치기 전에 말하기
- 데이터·큰 결과 파일은 커밋하지 않는다 (`.gitignore`)
- 커밋 메시지 예: `data: item_features 생성`, `cbf: k 튜닝 추가`
