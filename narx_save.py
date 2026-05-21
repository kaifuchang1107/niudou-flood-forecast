"""
訓練 NARX 直接法並儲存模型檔案
執行一次即可，產生 models/ 資料夾供 app.py 載入
"""
import pandas as pd
import numpy as np
import os, copy, warnings, joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

EVENT_DIR = r'C:\Users\USER\Desktop\KAIFU-Agent\修課\高等水文分析\final projet\事件劃分'
OUT_DIR   = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_KEYS = ['01_Nesat_Haitang','02_Maria','03_Mangkhut','04_Lekima',
              '05_Bailu','06_Vongfong','07_Chanthu','08_In-Fa',
              '11_Haikui','12_Koinu']
VAL_KEYS   = ['13_Gaemi','14_Ragasa']
H_LIST     = list(range(1, 7))
HIDDEN     = 16
PATIENCE   = 30
MAX_EPOCHS = 500
COMMON_START = 4

def load(key):
    df = pd.read_csv(os.path.join(EVENT_DIR, f'{key}.csv'),
                     parse_dates=['time'], index_col='time')
    L_start = df['L_m'].dropna().iloc[0]
    df['y']  = df['L_m'] - L_start
    df['L_start'] = L_start
    return df

dfs_train = {k: load(k) for k in TRAIN_KEYS}
dfs_val   = {k: load(k) for k in VAL_KEYS}

def build_direct_samples(dfs, h):
    Xs, Ys = [], []
    for df in dfs.values():
        y = df['y'].values; L = df['L_m'].values
        P = df['P_mm'].values; n = len(y)
        L_start = float(df['L_start'].iloc[0])
        for t in range(COMMON_START, n - h):
            if np.isnan(L[t+h]) or np.isnan(y[t]) or np.isnan(y[t-1]): continue
            row = [y[t], y[t-1], P[t]]
            if any(np.isnan(v) for v in row): continue
            Xs.append(row); Ys.append(L[t+h] - L_start)
    return np.array(Xs), np.array(Ys)

def train_es(X_tr, Y_tr, X_vl, Y_vl, tag=''):
    mlp = MLPRegressor(
        hidden_layer_sizes=(HIDDEN,), activation='tanh', solver='adam',
        learning_rate_init=0.001, batch_size=32,
        max_iter=1, warm_start=True, alpha=1e-4, random_state=42,
    )
    best_val = np.inf; no_imp = 0; best_mlp = None
    for epoch in range(1, MAX_EPOCHS+1):
        mlp.fit(X_tr, Y_tr)
        vl = np.mean((Y_vl - mlp.predict(X_vl))**2)
        if vl < best_val - 1e-6:
            best_val = vl; no_imp = 0; best_mlp = copy.deepcopy(mlp)
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f'  {tag} stop@{epoch}, val_mse={best_val:.5f}')
                return best_mlp
    print(f'  {tag} done@{MAX_EPOCHS}, val_mse={best_val:.5f}')
    return best_mlp

sigma_dict = {}

print('=== 訓練 NARX 直接法（各 h 獨立）===')
for h in H_LIST:
    Xt, Yt = build_direct_samples(dfs_train, h)
    Xv, Yv = build_direct_samples(dfs_val,   h)

    sx = StandardScaler().fit(Xt)
    sy = StandardScaler().fit(Yt.reshape(-1,1))

    mlp = train_es(sx.transform(Xt), sy.transform(Yt.reshape(-1,1)).ravel(),
                   sx.transform(Xv), sy.transform(Yv.reshape(-1,1)).ravel(),
                   tag=f'+{h}h')

    # 率定集 RMSE 作為 sigma 估計（與 ARX 一致）
    Yt_hat = sy.inverse_transform(mlp.predict(sx.transform(Xt)).reshape(-1,1)).ravel()
    sigma = float(np.sqrt(np.mean((Yt - Yt_hat)**2)))
    sigma_dict[h] = round(sigma, 4)
    print(f'  +{h}h  val RMSE = {sigma:.4f} m')

    joblib.dump(mlp, os.path.join(OUT_DIR, f'narx_d_h{h}.joblib'))
    joblib.dump(sx,  os.path.join(OUT_DIR, f'sx_d_h{h}.joblib'))
    joblib.dump(sy,  os.path.join(OUT_DIR, f'sy_d_h{h}.joblib'))

print(f'\nSIGMA_NARX_DIRECT = {sigma_dict}')
joblib.dump(sigma_dict, os.path.join(OUT_DIR, 'sigma_narx_direct.joblib'))
print(f'模型已存至 {OUT_DIR}')
