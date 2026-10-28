"""C1·C2 실행 스크립트 (notebooks/04_baselines_eval.ipynb와 같은 내용).

1) 기준 모델 4개를 train으로 fit → valid·test 예측값을 predictions/에 저장
2) 평점 지표(RMSE·MAE, 범위 × 구간), 랭킹 지표(①②③ HR@10·NDCG@10)
3) 성공 기준 ①(기준 모델 대비) 판정 예시, 결과 표를 results/baselines/에 저장
사용: python scripts/run_baselines.py [Video_Games Toys_and_Games]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from config import DATASETS, RESULT_DIR
from src.models.baselines import BASELINES
from src.eval import metrics as M

OUT = f'{RESULT_DIR}/baselines'
os.makedirs(OUT, exist_ok=True)


def run(ds: str):
    t0 = time.time()
    data = M.load_data(ds)
    print(f'=== {ds}: users={data.n_users:,} items={data.n_items:,} train={len(data.train):,} ({time.time()-t0:.0f}s)')
    models = {name: cls().fit(data.train, data.n_users, data.n_items) for name, cls in BASELINES.items()}

    # ---- 1) 예측값 저장 (valid, test)
    preds = {}
    for split in ['valid', 'test']:
        df = getattr(data, split)
        preds[split] = {name: m.predict(df['uidx'].values, df['iidx'].values) for name, m in models.items()}
        for name, p in preds[split].items():
            M.save_predictions(ds, name, split, df, p)
    print(f'  예측값 저장 완료 ({time.time()-t0:.0f}s)')

    # ---- 2) 평점 지표 (test)
    rt = M.rating_table(preds['test'], data.test, data)
    rt.insert(0, 'dataset', ds)
    rt.to_csv(f'{OUT}/{ds}_rating_test.csv', index=False)
    print('  [평점 RMSE, scope=all, group=all]')
    print(rt[(rt.scope == 'all') & (rt.group == 'all')][['model', 'n', 'rmse', 'mae']].to_string(index=False))

    # ---- 3) 랭킹 지표 (test): ①②는 전체 사용자 + p3 사용자, ③은 p3 사용자
    p3 = M.load_p3_users(ds)
    rk_all, _ = M.ranking_eval(models, data, 'test', protocols=('p1', 'p2'))
    rk_all['users'] = 'all'
    rk_p3, hits = M.ranking_eval(models, data, 'test', protocols=('p1', 'p2', 'p3'), users=p3)
    rk_p3['users'] = 'p3_users'
    rk = pd.concat([rk_all, rk_p3], ignore_index=True)
    rk.insert(0, 'dataset', ds)
    rk.to_csv(f'{OUT}/{ds}_ranking_test.csv', index=False)
    print('  [랭킹 HR@10, p3_users 공통 표본]')
    print(rk_p3.pivot_table(index='model', columns='protocol', values='HR10').round(4).to_string())

    # ---- 4) 성공 기준 ① 예시: 상품 평균을 "모델"로 놓고 나머지 기준 모델과 비교 (함수 동작 확인용)
    bl = {k: preds['test'][k] for k in ['global_mean', 'user_mean', 'item_mean']}
    c1 = M.criterion_rating_vs_baselines(preds['test']['item_mean'], bl, data.test, data, scope='all')
    print('  [기준 ① 함수 예시: item_mean vs baselines]'); print(c1.round(4).to_string(index=False))
    # 기준 ③ 예시: user_mean(사용자별 상수 점수)이 인기도를 이기는지 → 당연히 실패해야 정상
    c3 = M.criterion_topn_vs_popularity(hits, hybrid='user_mean', popularity='popularity', protocol='p3')
    print('  [기준 ③ 함수 예시: user_mean vs popularity on p3]', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c3.items()})
    M.save_json({'criterion1_example': c1.to_dict('records'), 'criterion3_example': c3}, f'{OUT}/{ds}_criteria_example.json')
    print(f'  완료 ({time.time()-t0:.0f}s)')
    return rt, rk


if __name__ == '__main__':
    for ds in (sys.argv[1:] or DATASETS):
        run(ds)
