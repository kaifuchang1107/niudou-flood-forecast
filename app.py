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
SIGMA_NARX   = {1:0.1725,2:0.2715,3:0.3430,4:0.4459,5:0.5425,6:0.5758}
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

    # 歷史水位
    if history:
        h_times = [x[0] for x in history]
        h_levels = [x[1] for x in history]
        fig.add_trace(go.Scatter(x=h_times, y=h_levels, mode='lines',
            name='觀測水位', line=dict(color='black', width=2.5)))

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
        xaxis_title='時間', yaxis_title='水位 (m)',
        legend=dict(x=0.01,y=0.99,bgcolor='rgba(255,255,255,0.8)'),
        height=480, margin=dict(l=60,r=120,t=50,b=50),
        hovermode='x unified',
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
    st.session_state.history = []
if 'last_fetch' not in st.session_state:
    st.session_state.last_fetch = None
if 'L_start_event' not in st.session_state:
    st.session_state.L_start_event = None   # 事件起始水位（鎖定後不變）
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

        # 更新歷史水位
        now_dt = datetime.fromisoformat(wl_time.replace('+08:00',''))
        if not st.session_state.history or st.session_state.history[-1][0] != now_dt:
            st.session_state.history.append((now_dt, L_now))
            st.session_state.history = st.session_state.history[-48:]

        # ── L_start 事件偵測（鎖定起始水位）────────────────────────
        if P_obs >= EVENT_P_THRESHOLD:
            if not st.session_state.event_active:
                # 事件開始：鎖定當前水位為 L_start
                st.session_state.L_start_event = L_now
                st.session_state.event_active  = True
        else:
            if st.session_state.event_active:
                # 降雨停止：重置（下次有雨再重新鎖定）
                st.session_state.event_active  = False
                st.session_state.L_start_event = None

        L_start_use = st.session_state.L_start_event if st.session_state.event_active else L_now

        rec, drc, nrx = run_forecast(L_now, P_obs, P_qpf, L_start=L_start_use)
        fetch_ok = True
    except Exception as e:
        st.error(f'資料抓取失敗：{e}')
        fetch_ok = False

if fetch_ok:
    # ── 狀態指標列 ─────────────────────────────────────────────
    prob1_2h = drc[1]['P_1']
    prob2_5h = drc[4]['P_2']

    if prob1_2h >= 30:
        status_text, status_color = '🚨 一級警戒風險', 'red'
    elif prob2_5h >= 30:
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

    # 第二列：超越機率
    r2c1, r2c2 = st.columns(2)
    r2c1.metric('+2h 超越一級警戒 (208.1m)', f'{prob1_2h:.1f}%')
    r2c2.metric('+5h 超越二級警戒 (206.8m)', f'{prob2_5h:.1f}%')

    st.markdown(f'### 防汛狀態：<span style="color:{status_color}">{status_text}</span>',
                unsafe_allow_html=True)

    # ── 主圖 ───────────────────────────────────────────────────
    fig = make_chart(L_now, wl_time, rec, drc, nrx, st.session_state.history)
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
- **二級警戒 206.8m**：+5h 直接法超越機率 ≥ 30% 時啟動。救災與防汛單位開始動員，完成人員、機具及防汛塊等搶險物資的整備與待命。
- **一級警戒 208.1m**：+2h 遞推法超越機率 ≥ 30% 時啟動。情況最為緊急，水情極度接近防洪上限；政府依《災害防救法》執行勸告或強制民眾撤離。

**不確定性**：95% CI 以各方法驗證集 RMSE 估計（ARX 用率定集，NARX 用驗證集）。
""")

# ── 自動重新整理 ────────────────────────────────────────────────
st.caption(f'API資料來源：水利署 OpenData (2560H024) | 氣象署 CWA (O-A0002-001, F-B0046-001)')
st.caption('第4組：張凱傅、蔡愷聆、張庭瑀')
time.sleep(REFRESH_SEC)
st.rerun()
