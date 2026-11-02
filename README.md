# Team 10 — 리뷰 기반 이커머스 추천 시스템과 평점 조작 탐지

2026-2 머신러닝 텀프로젝트 (정옥란 교수님) · 김율이, 배혜림, 최지훈

- 구축용 데이터: Amazon Reviews 2023 `Video_Games` (5-core)
- 테스트용 데이터: Amazon Reviews 2023 `Toys_and_Games` (5-core)
- 분할: 데이터셋에서 제공하는 leave-last-out 공식 분할
- 방식: 행렬 분해(MF), 콘텐츠 기반(TF-IDF), 하이브리드(평점 예측·Top-N 두 트랙), 가짜 프로필 탐지

## 폴더 구조

```
team10-recsys/
├── config.py                    # 경로·SEED·데이터셋 이름 (모두 이 파일을 import)
├── requirements.txt
├── docs/
│   ├── CONVENTIONS.md           # 팀 공통 규칙 (파일 형식, 모델 형식, 후보 파일, 결과물 위치)
│   └── EVAL_RULES.md            # 평가 규칙 (신뢰구간, 점수 주고받는 형식, 성공 기준 판정)
├── notebooks/                   # 실행용 Colab 노트북 (번호 순서대로 실행)
│   ├── 01_data_prep.ipynb       # 다운로드 → 검증 → 스키마 통일 → item_features (A1~A2)
│   ├── 02_eda.ipynb             # EDA 그래프 5개 + 요약표 (A3)
│   ├── 03_sampling.ipynb        # 프로토콜 ①②③ 후보 파일 + 점검 (A4)
│   ├── 04_baselines_eval.ipynb  # 기준 모델 + 공통 지표 (C1~C2)
│   ├── 05_mf.ipynb              # 행렬 분해 (B1~B3)
│   ├── 06_attack.ipynb          # 공격 주입·탐지·방어 (D1~D7)
│   ├── 07_cbf.ipynb             # 콘텐츠 기반 필터링 (B4)
│   ├── 08_hybrid.ipynb          # 하이브리드 평점 트랙·Top-N 트랙 (B5~B7)
│   └── runs/                    # 실행 결과(출력)가 저장된 노트북 사본. 코드 수정은 위 원본에서
│       ├── 07_cbf_Video_Games_run.ipynb
│       ├── 07_cbf_Toys_run.ipynb
│       ├── 08_hybrid_Video_Games_run.ipynb
│       └── 08_hybrid_Toys_run.ipynb
├── scripts/
│   └── run_baselines.py         # 04 노트북과 같은 내용의 실행 스크립트
├── src/
│   ├── data/
│   │   └── sampling.py          # 프로토콜 ①②③ 후보 샘플링 (A4)
│   ├── models/
│   │   ├── interface.py         # predict / score 공통 형식
│   │   ├── baselines.py         # 기준 모델 4개: 전역·사용자·상품 평균, 인기도 (C1)
│   │   ├── mf.py                # 행렬 분해, 편향 + L2 정규화 + SGD (B1~B3)
│   │   ├── cbf.py               # TF-IDF 콘텐츠 기반 필터링 (B4)
│   │   └── hybrid.py            # 하이브리드 두 트랙: 가중 결합, z-점수 랭킹 (B5~B7)
│   ├── eval/
│   │   ├── ranking.py           # 동점 무작위 처리 정답 순위 (target_rank, target_rank_full)
│   │   └── metrics.py           # 지표, 부트스트랩, 성공 기준 판정 함수 (C2)
│   └── attack/
│       ├── attacks.py           # 가짜 프로필 주입: random, average, bandwagon (D1~D2)
│       ├── detection.py         # 탐지 피처, 의사결정나무·kNN·k-means (D3~D6)
│       └── impact.py            # 공격 효과·회복 측정 (D7)
└── results/                     # 작은 그래프·표만 (큰 파일은 Drive)
```

## 데이터·결과 위치 (Google Drive)

데이터와 결과 파일은 GitHub에 올리지 않고 Drive 공유 폴더 `Team10_RecSys/`에 둔다.
공유받은 폴더는 각자 **우클릭 → 정리 → 바로가기 추가 → 내 드라이브**를 해야 Colab에서 `/content/drive/MyDrive/Team10_RecSys`로 보인다.
데이터 파일의 컬럼 설명은 `Team10_RecSys/data/processed/README.md`, 결과물 이름 규칙은 `docs/CONVENTIONS.md` 참고.

```
Team10_RecSys/
├── data/
│   ├── raw/{Video_Games, Toys_and_Games}/          # 원본 (01)
│   └── processed/{Video_Games, Toys_and_Games}/    # train·valid·test, user_map·item_map, item_features,
│       │                                           # item_popularity, tfidf.npz (01, 03, 07)
│       └── candidates/                             # 프로토콜 ①② 후보, ③ 사용자 (03)
├── predictions/{Video_Games, Toys_and_Games}/      # 모델별 valid·test 예측값 (04, 05, 07, 08)
└── results/
    ├── eda/                                        # EDA 그래프·요약표 (02)
    ├── sampling_sanity_check.csv                   # 샘플링 점검표 (03)
    ├── baselines/                                  # 기준 모델 평가표 (04)
    ├── mf/                                         # MF 튜닝·구성요소 분석·설정 (05)
    ├── attack/                                     # 공격·탐지·방어 결과 (06)
    ├── cbf/                                        # CBF 설정·k 튜닝·그래프 (07)
    └── hybrid/                                     # 하이브리드 설정·판정·사용자별 hit (08)
```

## 시작하기 (Colab)

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/qhfl3644/team10-recsys.git /content/team10-recsys
import sys; sys.path.append('/content/team10-recsys')
from config import *
```

노트북은 GitHub 주소로 바로 열 수 있다. 예: `https://colab.research.google.com/github/qhfl3644/team10-recsys/blob/main/notebooks/07_cbf.ipynb`
레포가 업데이트되면 노트북을 GitHub 주소로 **새로 열어야** 새 코드가 쓰인다 (이미 열린 창은 예전 코드).

## 실행 순서

| 순서 | 노트북 | 필요한 것 |
|---|---|---|
| 1 | 01 → 02 → 03 | 없음 (01이 원본을 내려받음) |
| 2 | 04, 05 | 03까지의 결과 |
| 3 | 06 | 05의 MF 설정 |
| 4 | 07 → 08 | 03까지의 결과. 08은 07의 `cbf` 예측값과 설정이 필요. MF 예측값 파일이 없으면 08이 직접 만든다 |

**Toys 일반화 실행 (07·08)**: 0번 셀의 두 줄만 바꾼다. Video_Games에서 고른 설정을 그대로 적용한 결과가 기본이고, Toys에서 다시 고른 결과는 `*_retuned`로 같이 저장된다.

```python
DS = 'Toys_and_Games'
TRANSFER_FROM = 'Video_Games'
```

**실행 결과를 레포에 남길 때**: Colab의 **파일 → GitHub에 사본 저장**에서 파일 경로를 `notebooks/runs/{원래 이름}_{데이터셋}_run.ipynb`로 바꿔 저장한다. 원래 이름으로 저장하면 원본 노트북이 실행 설정과 출력으로 덮어써진다.

## 진행 현황 (11/02 기준, 레포 기준)

| 구분 | 업무 | 담당 | 상태 |
|---|---|---|---|
| A1~A4 | 데이터 수집, 스키마 통일, EDA, 분할·샘플링 프로토콜 | 최지훈 | 완료 |
| B1~B3 | 행렬 분해, 튜닝, 구성요소 분석 | 배혜림 | Video_Games 완료 |
| B4~B7 | 콘텐츠 기반, 하이브리드 두 트랙 | 최지훈 | Video_Games·Toys 완료 (`notebooks/runs/`) |
| C1~C6 | 기준 모델, 평가 지표, 콜드스타트·일반화 실험, 결과 통합 | 김율이 | C1~C2 완료, 결과 통합 진행 중 |
| D1~D7 | 공격 주입, 탐지, 회복 측정 | 배혜림 | Video_Games 완료, 수치 조정 중 |
| E2~E6 | 발표자료, 보고서, 아키텍처, 저장소, 출처 | 김율이 | 진행 중 |

## 일정

| 기간 | 내용 |
|---|---|
| 10/24 ~ 10/30 | 전처리·EDA 완료, 구축용 데이터셋에서 MF 동작 |
| 10/31 ~ 11/04 | 콘텐츠 기반·하이브리드 완료, 공격 주입 모듈 |
| 11/05 ~ 11/07 | 탐지, 회복 측정, 일반화 검증, 구성요소 분석 |
| 11/08 ~ 11/09 | 발표자료·보고서·저장소 정리, 제출 |

## 참고

Hou, Y. et al. Bridging Language and Items for Retrieval and Recommendation. Amazon Reviews 2023. https://amazon-reviews-2023.github.io/
