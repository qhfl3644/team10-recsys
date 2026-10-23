# Team 10 — 리뷰 기반 이커머스 추천 시스템과 평점 조작 탐지

2026-2 머신러닝 텀프로젝트 (정옥란 교수님) · 김율이, 배혜림, 최지훈

- 구축용 데이터: Amazon Reviews 2023 `Video_Games` (5-core)
- 테스트용 데이터: Amazon Reviews 2023 `Toys_and_Games` (5-core)
- 분할: 데이터셋에서 제공하는 leave-last-out 공식 분할
- 방식: 행렬 분해(MF), 콘텐츠 기반(TF-IDF), 하이브리드(평점 예측·Top-N 두 트랙), 가짜 프로필 탐지

## 폴더 구조

```
team10-recsys/
├── config.py          # 경로·SEED·데이터셋 이름 (모두 이 파일을 import)
├── requirements.txt
├── docs/CONVENTIONS.md  # 팀 공통 규칙
├── notebooks/         # 실행용 Colab 노트북
│   └── 01_data_prep.ipynb   # 다운로드 → 검증 → 스키마 통일 → item_features
├── src/
│   ├── data/          # 전처리·EDA·샘플링 프로토콜 (A1~A4)
│   ├── models/        # 기준 모델, MF, CBF, 하이브리드 (B, C1)
│   │   └── interface.py     # predict / score 공통 형식
│   ├── eval/          # 지표, 프로토콜 ①②③, 동점 처리, 부트스트랩 (C)
│   └── attack/        # 공격 주입·탐지·회복 측정 (D)
└── results/           # 작은 그래프·표만 (큰 파일은 Drive)
```

## 데이터 위치

데이터는 GitHub에 올리지 않고 Google Drive 공유 폴더 `Team10_RecSys/`에 둔다.
파일 목록과 컬럼 설명은 `Team10_RecSys/data/processed/README.md` 참고.

```
Team10_RecSys/
├── data/raw/{Video_Games, Toys_and_Games}/
├── data/processed/{Video_Games, Toys_and_Games}/
├── predictions/
└── results/
```

## 시작하기 (Colab)

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/qhfl3644/team10-recsys.git /content/team10-recsys
import sys; sys.path.append('/content/team10-recsys')
from config import *
```

## 담당

| 구분 | 업무 | 담당 |
|---|---|---|
| A1~A4 | 데이터 수집, 스키마 통일, EDA, 분할·샘플링 프로토콜 | 최지훈 |
| B1~B3 | 행렬 분해, 튜닝, 구성요소 분석 | 배혜림 |
| B4~B7 | 콘텐츠 기반, 하이브리드 두 트랙 | 최지훈 |
| C1~C6 | 기준 모델, 평가 지표, 콜드스타트·일반화 실험, 결과 통합 | 김율이 |
| D1~D7 | 공격 주입, 탐지, 회복 측정 | 배혜림 |
| E2~E6 | 발표자료, 보고서, 아키텍처, 저장소, 출처 | 김율이 |

## 일정

| 기간 | 내용 |
|---|---|
| 10/24 ~ 10/30 | 전처리·EDA 완료, 구축용 데이터셋에서 MF 동작 |
| 10/31 ~ 11/04 | 콘텐츠 기반·하이브리드 완료, 공격 주입 모듈 |
| 11/05 ~ 11/07 | 탐지, 회복 측정, 일반화 검증, 구성요소 분석 |
| 11/08 ~ 11/09 | 발표자료·보고서·저장소 정리, 제출 |

## 참고

Hou, Y. et al. Bridging Language and Items for Retrieval and Recommendation. Amazon Reviews 2023. https://amazon-reviews-2023.github.io/
