"""팀 공통 설정. 경로·시드·데이터셋 이름은 이 파일에서만 바꾼다.

Colab 사용 예:
    from google.colab import drive; drive.mount('/content/drive')
    import sys; sys.path.append('/content/team10-recsys')   # git clone 한 위치
    from config import *
"""
import os

# Drive 공유 폴더 (팀원 각자 '바로가기 추가' 후 이 경로로 보이는지 확인)
ROOT = os.environ.get('TEAM10_ROOT', '/content/drive/MyDrive/Team10_RecSys')

RAW_DIR = f'{ROOT}/data/raw'                # 원본 (데이터셋별 하위 폴더)
PROC_DIR = f'{ROOT}/data/processed'         # 정리된 평점·상품 파일
PRED_DIR = f'{ROOT}/predictions'            # 모델 간 예측값 주고받는 곳
RESULT_DIR = f'{ROOT}/results'              # 큰 결과 파일 (작은 그래프·표는 레포 results/)

DATASETS = ['Video_Games', 'Toys_and_Games']    # [구축용, 테스트용]
SEED = 42
RATING_MIN, RATING_MAX = 1.0, 5.0


def proc_path(dataset: str, name: str) -> str:
    """예: proc_path('Video_Games', 'train') -> .../processed/Video_Games/train.parquet"""
    return f'{PROC_DIR}/{dataset}/{name}.parquet'


def pred_path(dataset: str, model: str, split: str) -> str:
    """예: pred_path('Video_Games', 'mf', 'valid') -> .../predictions/Video_Games/mf_valid.parquet"""
    return f'{PRED_DIR}/{dataset}/{model}_{split}.parquet'
