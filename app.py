"""
蘭陽溪牛鬥(3) 即時洪水預報系統 — Streamlit 網頁介面
====================================================
執行方式：streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import requests
import warnings
from datetime import datetime, timedelta
from scipy import stats
import time, os, joblib

# ── Google Sheets 寫入（可選，失敗不影響主功能）─────────────────
def sheets_append(entry: dict):
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        creds  = Credentials.from_service_account_info(
            dict(st.secrets['gcp_service_account']), scopes=scopes)
        client = gspread.authorize(creds)
        ws = client.open_by_key(st.secrets['SHEET_ID']).sheet1
        # 第一次寫入時建立標題列
        if ws.acell('A1').value != '時間':
            ws.insert_row(list(entry.keys()), 1)
        ws.append_row(list(entry.values()), value_input_option='USER_ENTERED')
    except Exception:
        pass  # Sheets 失敗不影響主功能

warnings.filterwarnings('ignore')

# ── 頁面設定 ────────────────────────────────────────────────────
st.set_page_config(
    page_title='牛鬥(3) 即時洪水預報',
    page_icon='🌊',
    layout='wide',
)

# ── 常數 ─────────────────────────────────────────────────────────
CWA_KEY    = st.secrets["CWA_KEY"]
WL_STATION = '2560H024'
WARNING    = {'二級警戒': 206.8, '一級警戒': 208.1}
THIESSEN   = {'C0U720':0.345,'C0UA00':0.182,'C0UA30':0.141,
              'C1U501':0.125,'C0U710':0.121,'C0UA10':0.085}
STATION_NAME = {'C0U720':'南山','C0UA00':'土場','C0UA30':'白嶺',
                'C1U501':'牛鬥','C0U710':'太平山','C0UA10':'鴛鴦湖'}
STATION_COORDS = {
    'C0U720':(24.439167,121.373333), 'C0UA00':(24.578356,121.487153),
    'C0UA30':(24.529611,121.510844), 'C1U501':(24.639444,121.565278),
    'C0U710':(24.507222,121.517500), 'C0UA10':(24.592053,121.404039),
}

PHI1, PHI2, B1 = 1.34380, -0.36121, 0.005470
SIGMA_RECUR = {1:0.0861,2:0.1477,3:0.1956,4:0.2350,5:0.2671,6:0.2954}
DIRECT_COEF = {
    1:(1.343802,-0.361208,0.005470), 2:(1.412982,-0.449921,0.011151),
    3:(1.437064,-0.488761,0.014544), 4:(1.404450,-0.471521,0.018064),
    5:(1.418769,-0.499015,0.020420), 6:(1.379849,-0.474971,0.023350),
}
SIGMA_DIRECT = {1:0.0861,2:0.1516,3:0.2078,4:0.2565,5:0.2990,6:0.3383}
SIGMA_NARX   = {1:0.0764,2:0.1219,3:0.1648,4:0.2038,5:0.2443,6:0.2784}
Z95 = 1.96
QPF_LON0,QPF_LAT0,QPF_RES,QPF_NX,QPF_NY = 117.975,19.975,0.0125,441,561
REFRESH_SEC = 600  # 10 分鐘自動更新

# ── NARX 模型載入（快取，只載入一次）────────────────────────────────
@st.cache_resource
def load_narx():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    m = {}
    for h in range(1, 7):
        m[h] = {
            'mlp': joblib.load(os.path.join(base, f'narx_d_h{h}.joblib')),
            'sx':  joblib.load(os.path.join(base, f'sx_d_h{h}.joblib')),
            'sy':  joblib.load(os.path.join(base, f'sy_d_h{h}.joblib')),
        }
    return m

# ── 資料抓取函式 ──────────────────────────────────────────────────
@st.cache_data(ttl=REFRESH_SEC)
def fetch_all():
    """抓水位、觀測雨量、QPF，回傳 dict"""
    result = {}

    # 1. 水位
    url = ('https://opendata.wra.gov.tw/api/v2/'
           '73c4c3de-4045-4765-abeb-89f9f9cd5ff0'
           '?sort=_importdate asc&format=JSON')
    data = requests.get(url, timeout=15, verify=False).json()
    for row in data:
        if row['stationid'] == WL_STATION:
            result['L_now'] = float(row['waterlevel'])
            result['wl_time'] = row['datetime']
            break

    # 2. Thiessen 觀測雨量
    url2 = (f'https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0002-001'
            f'?Authorization={CWA_KEY}')
    stations = requests.get(url2, timeout=15, verify=False).json()['records']['Station']
    P_obs = 0.0; rain_detail = {}
    for s in stations:
        sid = s.get('StationId','')
        if sid in THIESSEN:
            p = float(s['RainfallElement']['Past1hr']['Precipitation'] or 0)
            P_obs += THIESSEN[sid] * p
            rain_detail[STATION_NAME[sid]] = p
    result['P_obs'] = P_obs
    result['rain_detail'] = rain_detail
    # 用水利署資料時間（已含+08:00 台灣時間）作為更新時間
    result['obs_time'] = result.get('wl_time', '')

    # 3. QPF
    url3 = (f'https://opendata.cwa.gov.tw/fileapi/v1/opendataapi/F-B0046-001'
            f'?Authorization={CWA_KEY}&format=JSON')
    d = requests.get(url3, timeout=20, verify=False).json()['cwaopendata']['dataset']
    result['qpf_time'] = d['datasetInfo']['parameterSet']['DateTime']
    vals = np.array([float(x) for x in d['contents']['content'].split(',')])
    grid = vals.reshape(QPF_NY, QPF_NX)
    P_qpf = 0.0; valid = 0
    for sid, w in THIESSEN.items():
        if sid not in STATION_COORDS: continue
        lat, lon = STATION_COORDS[sid]
        ix = int(round((lon-QPF_LON0)/QPF_RES))
        iy = int(round((lat-QPF_LAT0)/QPF_RES))
        ix = max(0,min(ix,QPF_NX-1)); iy = max(0,min(iy,QPF_NY-1))
        v = grid[iy,ix]
        if v != -99.0:
            P_qpf += w*v; valid += 1
    result['P_qpf'] = P_qpf if valid > 0 else None

    return result

# ── 預測函式 ──────────────────────────────────────────────────────
def run_forecast(L_now, P_obs, P_qpf, L_start=None):
    if L_start is None:
        L_start = L_now   # 無事件：滾動基準
    y_cur  = L_now - L_start
    y_prev = L_now - L_start   # L_prev 仍用 L_now 近似

    # 遞推（+1h, +2h）
    P_next = P_qpf if P_qpf is not None else P_obs
    rec = []
    buf = [y_cur, y_cur]   # 用 y_cur 初始化兩步（L_prev ≈ L_now）
    for h in range(1, 3):
        P_use = P_obs if h == 1 else P_next
        nd = PHI1*buf[-1] + PHI2*buf[-2] + B1*P_use
        buf.append(nd)
        L_hat = nd + L_start
        sig = SIGMA_RECUR[h]
        rec.append({'h':h,'L_hat':L_hat,'lo':L_hat-Z95*sig,'hi':L_hat+Z95*sig,'sig':sig})

    # ARX 直接法（+1h~+6h）
    drc = []
    for h in range(1, 7):
        a,b,c = DIRECT_COEF[h]
        L_hat = a*y_cur + b*y_prev + c*P_obs + L_start
        sig = SIGMA_DIRECT[h]
        p2 = (1-stats.norm.cdf((WARNING['二級警戒']-L_hat)/sig))*100
        p1 = (1-stats.norm.cdf((WARNING['一級警戒']-L_hat)/sig))*100
        drc.append({'h':h,'L_hat':L_hat,'lo':L_hat-Z95*sig,'hi':L_hat+Z95*sig,
                    'sig':sig,'P_2':p2,'P_1':p1})

    # NARX 直接法（+1h~+6h）
    narx_m = load_narx()
    nrx = []
    for h in range(1, 7):
        m  = narx_m[h]
        xsc = m['sx'].transform([[y_cur, y_prev, P_obs]])
        y_hat = float(m['sy'].inverse_transform(
            m['mlp'].predict(xsc).reshape(-1,1))[0][0])
        L_hat = y_hat + L_start
        sig = SIGMA_NARX[h]
        p2 = (1-stats.norm.cdf((WARNING['二級警戒']-L_hat)/sig))*100
        p1 = (1-stats.norm.cdf((WARNING['一級警戒']-L_hat)/sig))*100
        nrx.append({'h':h,'L_hat':L_hat,'lo':L_hat-Z95*sig,'hi':L_hat+Z95*sig,
                    'sig':sig,'P_2':p2,'P_1':p1})

    return rec, drc, nrx

# ── Plotly 歷線圖 ─────────────────────────────────────────────────
def make_chart(L_now, wl_time, rec, drc, nrx, history):
    fig = go.Figure()
    now_dt = datetime.fromisoformat(wl_time.replace('+08:00',''))
    t_future = [now_dt + timedelta(hours=h) for h in range(1,7)]
    t_rec   = [now_dt + timedelta(hours=h) for h in range(1,3)]

    # 歷史降雨（右 y 軸，倒置）
    if history:
        h_times  = [x[0] for x in history]
        h_levels = [x[1] for x in history]
        h_rain   = [x[2] if len(x) > 2 else 0 for x in history]
        max_rain = max(h_rain) if max(h_rain) > 0 else 10

        fig.add_trace(go.Bar(
            x=h_times, y=h_rain, name='觀測雨量 (mm/h)',
            marker_color='cornflowerblue', opacity=0.35,
            yaxis='y2'
        ))
        fig.add_trace(go.Scatter(
            x=h_times, y=h_levels, mode='lines',
            name='觀測水位', line=dict(color='black', width=2.5)
        ))
    else:
        max_rain = 10

    # 當前點
    fig.add_trace(go.Scatter(x=[now_dt], y=[L_now], mode='markers',
        name=f'當前 {L_now:.3f}m', marker=dict(color='black',size=10,symbol='circle')))

    # 直接法 CI 帶（+1h~+6h）
    drc_lo = [d['lo'] for d in drc]
    drc_hi = [d['hi'] for d in drc]
    drc_y  = [d['L_hat'] for d in drc]
    fig.add_trace(go.Scatter(
        x=t_future+t_future[::-1], y=drc_hi+drc_lo[::-1],
        fill='toself', fillcolor='rgba(70,130,180,0.15)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=True,
        name='直接法 95% CI'))
    fig.add_trace(go.Scatter(x=t_future, y=drc_y, mode='lines+markers',
        name='直接法 +1h~+6h', line=dict(color='steelblue',width=2,dash='dot'),
        marker=dict(size=7)))

    # NARX 直接法 CI 帶（+1h~+6h）
    nrx_lo = [n['lo'] for n in nrx]
    nrx_hi = [n['hi'] for n in nrx]
    nrx_y  = [n['L_hat'] for n in nrx]
    fig.add_trace(go.Scatter(
        x=t_future+t_future[::-1], y=nrx_hi+nrx_lo[::-1],
        fill='toself', fillcolor='rgba(39,174,96,0.12)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=True,
        name='NARX直接 95% CI'))
    fig.add_trace(go.Scatter(x=t_future, y=nrx_y, mode='lines+markers',
        name='NARX直接 +1h~+6h', line=dict(color='seagreen',width=2,dash='dashdot'),
        marker=dict(size=7, symbol='diamond')))

    # 遞推 + QPF（+1h, +2h）
    rec_lo = [r['lo'] for r in rec]
    rec_hi = [r['hi'] for r in rec]
    rec_y  = [r['L_hat'] for r in rec]
    rec_full_x = [now_dt] + t_rec
    rec_full_y = [L_now]  + rec_y
    fig.add_trace(go.Scatter(x=t_rec+t_rec[::-1], y=rec_hi+rec_lo[::-1],
        fill='toself', fillcolor='rgba(220,80,60,0.15)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=True, name='遞推 95% CI'))
    fig.add_trace(go.Scatter(x=rec_full_x, y=rec_full_y, mode='lines+markers',
        name='遞推+QPESUMS +1h~+2h', line=dict(color='tomato',width=2.5),
        marker=dict(size=8)))

    # 警戒水位
    for name, level in WARNING.items():
        color = 'darkorange' if '二級' in name else 'crimson'
        dash  = 'dash' if '二級' in name else 'solid'
        fig.add_hline(y=level, line_color=color, line_dash=dash, line_width=1.5,
                      annotation_text=f'{name} {level}m',
                      annotation_position='right')

    fig.update_layout(
        title=dict(text='蘭陽溪牛鬥(3) 即時水位預報', font=dict(size=16)),
        xaxis_title='時間',
        yaxis=dict(title='水位 (m)'),
        yaxis2=dict(
            title='雨量 (mm/h)',
            overlaying='y', side='right',
            range=[max(max_rain * 4, 20), 0],
            tickfont=dict(color='cornflowerblue'),
            title_font=dict(color='cornflowerblue'),
            showgrid=False,
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        height=500, margin=dict(l=60, r=80, t=50, b=50),
        hovermode='x unified',
        barmode='overlay',
    )
    return fig

# ── Sidebar ───────────────────────────────────────────────────────
IMG_PATH = '牛鬥橋_Niudou_Bridge_-_panoramio.jpg'
with st.sidebar:
    try:
        st.image(IMG_PATH, use_container_width=True)
    except Exception:
        pass
    st.caption('牛鬥橋 Niudou Bridge\n蘭陽溪，宜蘭縣大同鄉')
    st.markdown('---')
    st.markdown('''
**監測站**：牛鬥(3)水位站

**集水區面積**：約455km²

**警戒水位**（水利署）
- 二級：206.8 m
- 一級：208.1 m

---
**高等水文分析 第4組**
張凱傅、蔡愷聆、張庭瑀
''')
    st.markdown('---')
    if st.button('🔗 測試 Google Sheets 連線', use_container_width=True):
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            scopes = ['https://www.googleapis.com/auth/spreadsheets']
            creds  = Credentials.from_service_account_info(
                dict(st.secrets['gcp_service_account']), scopes=scopes)
            client = gspread.authorize(creds)
            ws = client.open_by_key(st.secrets['SHEET_ID']).sheet1
            st.success(f'✅ 連線正常（試算表：{ws.spreadsheet.title}）')
        except Exception as e:
            st.error(f'錯誤：{e}')

# ── 手機響應式 CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* 手機版：指標卡換行 */
@media (max-width: 768px) {
    [data-testid="column"] {
        min-width: 45% !important;
        flex: 1 1 45% !important;
    }
    /* 圖表全寬 */
    .js-plotly-plot { width: 100% !important; }
    /* 縮小標題 */
    h1 { font-size: 1.4rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ── 主介面 ────────────────────────────────────────────────────────
st.title('🌊 蘭陽溪牛鬥(3) 即時洪水預報系統')
st.caption('高等水文分析 第4組：張凱傅、蔡愷聆、張庭瑀')
st.caption('模型：ARX(2,1,0) 遞推/直接法 + NARX直接法 | 資料：水利署 + 氣象署 CWA | 每10分鐘自動更新')

# 初始化 session state
if 'history' not in st.session_state:
    st.session_state.history = []           # 滾動12小時 (time, L, P)
if 'event_history' not in st.session_state:
    st.session_state.event_history = []    # 完整事件歷線 (time, L, P)
if 'pred_log' not in st.session_state:
    st.session_state.pred_log = []          # 完整預測日誌（每次刷新一筆）
if 'last_fetch' not in st.session_state:
    st.session_state.last_fetch = None
if 'L_start_event' not in st.session_state:
    st.session_state.L_start_event = None
if 'event_active' not in st.session_state:
    st.session_state.event_active = False

EVENT_P_THRESHOLD = 5.0   # mm/h，超過此值視為洪水事件開始

# 手動刷新按鈕
col_btn, col_time = st.columns([1,4])
with col_btn:
    if st.button('🔄 立即更新', use_container_width=True):
        st.cache_data.clear()
with col_time:
    st.caption(f'自動更新間隔：10分鐘 | 資料時間（台灣）：{st.session_state.last_fetch or "—"}')

# 抓資料
with st.spinner('抓取即時資料中...'):
    try:
        data = fetch_all()
        L_now  = data['L_now']
        P_obs  = data['P_obs']
        P_qpf  = data['P_qpf']
        wl_time = data['wl_time']
        try:
            wl_dt = datetime.fromisoformat(data['wl_time'].replace('+08:00',''))
            st.session_state.last_fetch = wl_dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            st.session_state.last_fetch = data.get('wl_time','—')

        # 時間解析
        now_dt   = datetime.fromisoformat(wl_time.replace('+08:00',''))
        this_hour = now_dt.replace(minute=0, second=0, microsecond=0)

        # 整點判斷：history 最後一筆的整點 ≠ 現在的整點
        last_hour = (st.session_state.history[-1][0].replace(minute=0, second=0, microsecond=0)
                     if st.session_state.history else None)
        is_new_hour = (last_hour is None or last_hour != this_hour)

        # ── 整點寫入歷史（滾動 24 筆 = 24 小時）──────────────────────
        if is_new_hour:
            st.session_state.history.append((this_hour, L_now, P_obs))
            st.session_state.history = st.session_state.history[-24:]

        # ── L_start 事件偵測（鎖定起始水位）────────────────────────
        if P_obs >= EVENT_P_THRESHOLD:
            if not st.session_state.event_active:
                st.session_state.L_start_event = L_now
                st.session_state.event_active  = True
                st.session_state.event_history = []
            if is_new_hour:
                st.session_state.event_history.append((this_hour, L_now, P_obs))
        else:
            if st.session_state.event_active:
                if is_new_hour:
                    st.session_state.event_history.append((this_hour, L_now, P_obs))
                st.session_state.event_active  = False
                st.session_state.L_start_event = None

        L_start_use = st.session_state.L_start_event if st.session_state.event_active else L_now

        rec, drc, nrx = run_forecast(L_now, P_obs, P_qpf, L_start=L_start_use)

        # ── 預測日誌（整點記一筆）────────────────────────────────────
        if is_new_hour:
            entry = {
                '時間':           this_hour.strftime('%Y-%m-%d %H:%M'),
                '水位_obs(m)':    round(L_now, 3),
                '雨量_obs(mm/h)': round(P_obs, 2),
                'QPF(mm/h)':     round(P_qpf, 2) if P_qpf is not None else '',
                'ARX遞推_+1h':    round(rec[0]['L_hat'], 3),
                'ARX遞推_+2h':    round(rec[1]['L_hat'], 3),
            }
            for d in drc:
                entry[f'ARX直接_+{d["h"]}h'] = round(d['L_hat'], 3)
            for n in nrx:
                entry[f'NARX直接_+{n["h"]}h'] = round(n['L_hat'], 3)
            st.session_state.pred_log.append(entry)
            sheets_append(entry)   # 同步寫入 Google Sheets

        fetch_ok = True
    except Exception as e:
        st.error(f'資料抓取失敗：{e}')
        fetch_ok = False

if fetch_ok:
    # ── 狀態指標列 ─────────────────────────────────────────────
    # 觸發條件：95% CI 上界碰到警戒水位（更保守）
    ci_1_2h = rec[1]['hi']   # 遞推法 +2h CI 上界 vs 一級警戒 208.1m
    ci_2_5h = drc[4]['hi']   # 直接法 +5h CI 上界 vs 二級警戒 206.8m

    alert_1 = ci_1_2h >= WARNING['一級警戒']
    alert_2 = ci_2_5h >= WARNING['二級警戒']

    if alert_1:
        status_text, status_color = '🚨 一級警戒風險', 'red'
    elif alert_2:
        status_text, status_color = '⚠️ 二級警戒風險', 'orange'
    else:
        status_text, status_color = '✅ 安全', 'green'

    # 第一列：水位 + 雨量 + QPF
    try:
        qpf_dt = datetime.fromisoformat(data['qpf_time'].replace('+08:00',''))
        qpf_str = qpf_dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        qpf_str = data.get('qpf_time','—')

    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric('當前水位',
                f'{L_now:.3f} m',
                delta=f'{L_now-WARNING["二級警戒"]:.1f}m 距二級' if L_now < WARNING['二級警戒'] else '超過二級警戒!')
    r1c2.metric('觀測雨量 (Past1hr)', f'{P_obs:.2f} mm/h')
    r1c3.metric('QPESUMS 1小時定量降雨預報',
                f'{P_qpf:.2f} mm/h' if P_qpf is not None else '0.00 mm/h',
                help=f'來源：CWA F-B0046-001（雷達外推法），資料時間 {qpf_str}。')

    # 第二列：CI 上界 vs 警戒水位
    r2c1, r2c2 = st.columns(2)
    gap1 = ci_1_2h - WARNING['一級警戒']
    gap2 = ci_2_5h - WARNING['二級警戒']
    r2c1.metric('+2h 遞推 95%CI上界（一級警戒 208.1m）',
                f'{ci_1_2h:.3f} m',
                delta=f'{gap1:+.3f} m',
                delta_color='inverse')
    r2c2.metric('+5h 直接 95%CI上界（二級警戒 206.8m）',
                f'{ci_2_5h:.3f} m',
                delta=f'{gap2:+.3f} m',
                delta_color='inverse')

    st.markdown(f'### 防汛狀態：<span style="color:{status_color}">{status_text}</span>',
                unsafe_allow_html=True)

    # ── 主圖 ───────────────────────────────────────────────────
    # 事件中顯示完整事件歷線，否則顯示滾動12小時
    display_history = (st.session_state.event_history
                       if st.session_state.event_active and st.session_state.event_history
                       else st.session_state.history)
    n_hist = len(display_history)
    if n_hist > 1:
        t0 = display_history[0][0].strftime('%m/%d %H:%M')
        t1 = display_history[-1][0].strftime('%m/%d %H:%M')
        hist_label = f'歷史資料：{t0} ～ {t1}（{n_hist} 筆）'
    else:
        hist_label = '歷史資料：本次連線起始，累積中…'
    st.caption(hist_label)
    fig = make_chart(L_now, wl_time, rec, drc, nrx, display_history)
    st.plotly_chart(fig, use_container_width=True)

    # ── 預測數值表 ─────────────────────────────────────────────
    st.subheader('預測數值')
    col_r, col_d, col_n = st.columns(3)

    with col_r:
        st.markdown('**🔴 遞推法 + QPESUMS（+1h, +2h）**')
        rows = []
        for r in rec:
            qpf_tag = '(QPESUMS)' if r['h'] == 2 and P_qpf is not None else '(觀測)'
            rows.append({
                '預報時距': f'+{r["h"]}h {qpf_tag}',
                '預測水位 (m)': f'{r["L_hat"]:.3f}',
                '95% CI': f'[{r["lo"]:.3f}, {r["hi"]:.3f}]',
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)

    with col_d:
        st.markdown('**🔵 ARX直接法（+1h ~ +6h）**')
        rows2 = []
        for d in drc:
            warn = ''
            if d['P_1'] >= 30: warn = '🚨'
            elif d['P_2'] >= 30: warn = '⚠️'
            rows2.append({
                '預報時距': f'+{d["h"]}h',
                '預測水位 (m)': f'{d["L_hat"]:.3f}',
                '95% CI': f'[{d["lo"]:.3f}, {d["hi"]:.3f}]',
                'P(>206.8m)': f'{d["P_2"]:.1f}%',
                'P(>208.1m)': f'{d["P_1"]:.1f}%',
                '警示': warn,
            })
        st.dataframe(rows2, hide_index=True, use_container_width=True)

    with col_n:
        st.markdown('**🟢 NARX直接法（+1h ~ +6h）**')
        rows3 = []
        for n in nrx:
            warn = ''
            if n['P_1'] >= 30: warn = '🚨'
            elif n['P_2'] >= 30: warn = '⚠️'
            rows3.append({
                '預報時距': f'+{n["h"]}h',
                '預測水位 (m)': f'{n["L_hat"]:.3f}',
                '95% CI': f'[{n["lo"]:.3f}, {n["hi"]:.3f}]',
                'P(>206.8m)': f'{n["P_2"]:.1f}%',
                'P(>208.1m)': f'{n["P_1"]:.1f}%',
                '警示': warn,
            })
        st.dataframe(rows3, hide_index=True, use_container_width=True)

    # ── 下載區 ──────────────────────────────────────────────────
    import io, csv as _csv
    dl_col1, dl_col2 = st.columns(2)

    # 1. 預測日誌（完整，含三種模型輸出）
    with dl_col1:
        if st.session_state.pred_log:
            buf1 = io.StringIO()
            cols = list(st.session_state.pred_log[0].keys())
            w1 = _csv.DictWriter(buf1, fieldnames=cols)
            w1.writeheader(); w1.writerows(st.session_state.pred_log)
            st.download_button(
                label=f'⬇️ 預測日誌 CSV（{len(st.session_state.pred_log)} 筆）',
                data=buf1.getvalue().encode('utf-8-sig'),
                file_name=f'forecast_log_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                mime='text/csv',
                help='含水位觀測、雨量、QPF、ARX遞推/直接、NARX直接預測值',
            )
        else:
            st.caption('預測日誌累積中…')

    # 2. 事件/近期歷線（水位+雨量）
    with dl_col2:
        dl_data = st.session_state.event_history or st.session_state.history
        if dl_data:
            buf2 = io.StringIO()
            w2 = _csv.writer(buf2)
            w2.writerow(['time', 'L_m', 'P_mm_h'])
            for row in dl_data:
                w2.writerow([row[0].strftime('%Y-%m-%d %H:%M'),
                             f'{row[1]:.3f}', f'{row[2]:.2f}'])
            label = '事件歷線' if st.session_state.event_history else '近期歷線'
            st.download_button(
                label=f'⬇️ {label} CSV（{len(dl_data)} 筆）',
                data=buf2.getvalue().encode('utf-8-sig'),
                file_name=f'niudou_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                mime='text/csv',
            )

    # ── 雨量站詳情 ─────────────────────────────────────────────
    with st.expander('Thiessen 六站雨量詳情'):
        rain_rows = []
        for k, v in data['rain_detail'].items():
            sid = next((s for s, n in STATION_NAME.items() if n == k), '')
            w = THIESSEN.get(sid, 0)
            rain_rows.append({
                '站名': k,
                'Thiessen 權重': f'{w*100:.1f}%',
                '過去1小時雨量 (mm)': f'{v:.1f}',
                '加權貢獻 (mm)': f'{w*v:.3f}',
            })
        st.dataframe(rain_rows, hide_index=True, use_container_width=True)
        st.caption('資料來源：氣象署 CWA 自動雨量站（O-A0002-001），過去1小時觀測累積值')

    # ── 說明 ───────────────────────────────────────────────────
    with st.expander('模型說明'):
        st.markdown('#### 水位偏差定義')
        st.markdown('各預測方法皆以水位偏差 $y_t$ 為操作變數，事件起始水位 $L_{\\text{start}}$ 在降雨超過 5 mm/h 時鎖定：')
        st.latex(r'y_t = L_t - L_{\text{start}}')
        st.markdown('---')

        st.markdown('#### 🔴 ARX 遞推法（+1h, +2h）')
        st.markdown('線性 ARX(2,1,0) 模型，每步預測回饋至下一步；+2h 使用氣象署 QPESUMS 雷達外推雨量。')
        st.latex(r'\hat{y}_{t+1} = \varphi_1\,y_t + \varphi_2\,y_{t-1} + \beta_1\,P_t')
        st.latex(r'\hat{L}_{t+h} = \hat{y}_{t+h} + L_{\text{start}}')
        st.markdown('---')

        st.markdown('#### 🔵 ARX 直接法（+1h ~ +6h）')
        st.markdown('各預報時距 $h$ 獨立率定線性模型，僅用當前觀測，不回饋預測誤差，不依賴未來雨量。')
        st.latex(r'\hat{L}_{t+h} = a_h\,y_t + b_h\,y_{t-1} + c_h\,P_t + L_{\text{start}}')
        st.markdown('---')

        st.markdown('#### 🟢 NARX 直接法（+1h ~ +6h）')
        st.markdown('各預報時距 $h$ 獨立訓練單隱藏層神經網路，輸入與 ARX 相同，以非線性函數取代線性係數。')
        st.latex(r'\hat{L}_{t+h} = f_h\!\bigl(y_t,\,y_{t-1},\,P_t\bigr) + L_{\text{start}}')
        st.markdown('架構：3 → 16 (tanh) → 1 (linear)。')
        st.markdown('---')

        st.markdown('#### 防汛警戒對應（水利署定義）')
        st.markdown("""
- **二級警戒 206.8m**：+5h ARX直接法 **95% CI 上界** 碰到 206.8m 時啟動。救災與防汛單位開始動員，完成人員、機具及防汛塊等搶險物資的整備與待命。
- **一級警戒 208.1m**：+2h ARX遞推法 **95% CI 上界** 碰到 208.1m 時啟動。情況最為緊急，水情極度接近防洪上限；政府依《災害防救法》執行勸告或強制民眾撤離。

以 CI 上界作為觸發條件，較超越機率門檻更保守，可提前發出預警。

**不確定性**：95% CI 以率定集 RMSE 估計（三種方法一致）。
""")

# ── 自動重新整理 ────────────────────────────────────────────────
st.caption(f'API資料來源：水利署 OpenData (2560H024) | 氣象署 CWA (O-A0002-001, F-B0046-001)')
st.caption('第4組：張凱傅、蔡愷聆、張庭瑀')
time.sleep(REFRESH_SEC)
st.rerun()
