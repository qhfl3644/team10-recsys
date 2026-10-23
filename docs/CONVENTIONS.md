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

## 실험 규칙

- `SEED = 42` (numpy, random, 샘플링 모두)
- 하이퍼파라미터·가중치는 **valid로만** 선택, test는 마지막에 한 번만
- 랭킹 동점은 무작위로 순위 결정 (평가 담당 함수 사용)

## Git

- `main`에 바로 올려도 되지만, 남의 폴더 파일은 고치기 전에 말하기
- 데이터·큰 결과 파일은 커밋하지 않는다 (`.gitignore`)
- 커밋 메시지 예: `data: item_features 생성`, `cbf: k 튜닝 추가`
