"""
Ядро расчётов TDA для сайта: загрузка котировок, облако точек,
диаграммы персистентности (Vietoris–Rips), Wasserstein distance,
L1/L2 нормы persistence landscapes и пороги mu + k*sigma.
"""
import io
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform, cdist

try:
    from ripser import ripser as _ripser
except ImportError:  # H0 всё равно считается через MST, H1 требует ripser
    _ripser = None

BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"),
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.wsj.com/market-data",
}


# =============================================================================
# 1. ДАННЫЕ
# =============================================================================
def wsj_path(user_input: str) -> str:
    """'RUT' -> 'index/RUT'; 'HK/XHKG/HSI' -> 'index/HK/XHKG/HSI';
    полная ссылка WSJ -> путь между /quotes/ и /historical-prices."""
    s = user_input.strip()
    m = re.search(r"/quotes/(.+?)(?:/historical-prices.*)?$", s)
    if m and "wsj.com" in s:
        return m.group(1).strip("/")
    s = s.strip("/")
    if s.lower().startswith("index/"):
        return "index/" + s[6:].upper()
    return "index/" + s.upper()


def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next(c for c in df.columns if c.lower() in ("date", "datums", "дата"))
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]) and df[date_col].astype(str).str.match(r"^\d{1,2}\.\d{1,2}\.\d{2,4}").any():
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")  # 04.01.2016
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")  # WSJ: 11/14/26
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    out = pd.DataFrame(index=df.index)

    def pick(keys, exclude=()):
        # ключи по приоритету: сначала точное совпадение, затем подстрока
        for k in keys:
            for c in df.columns:
                if c.lower() == k:
                    return c
        for k in keys:
            for c in df.columns:
                cl = c.lower()
                if k in cl and not any(x in cl for x in exclude):
                    return c
        return None

    for target, keys, excl in [
        ("Open", ["open", "opening value", "opening"], ()),
        ("High", ["high"], ()),
        ("Low", ["low"], ()),
        ("Close", ["close", "closing", "index value", "value", "last"], ("open", "adj")),
    ]:
        col = pick(keys, excl)
        if col is not None:
            out[target] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.replace(" ", ""), errors="coerce")
    if "Close" not in out:
        raise ValueError(f"Не найдена колонка с ценой закрытия. Колонки: {list(df.columns)}")
    out.index.name = "Date"
    return out.dropna(subset=["Close"])


def fetch_wsj(user_input: str, start: date, end: date) -> pd.DataFrame:
    path = wsj_path(user_input)
    n = (end - start).days + 1
    url = (f"https://www.wsj.com/market-data/quotes/{path}/historical-prices/download"
           f"?MOD_VIEW=page&num_rows={n}&range_days={n}"
           f"&startDate={start:%m/%d/%Y}&endDate={end:%m/%d/%Y}")
    r = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
    r.raise_for_status()
    text = r.text
    if "date" not in text[:200].lower():
        raise ValueError("WSJ вернул не CSV (вероятно, блокировка бота или неверный тикер).")
    df = pd.read_csv(io.StringIO(text))
    return _clean_ohlc(df)


def fetch_yahoo(ticker: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf
    candidates = [ticker] if ticker.startswith("^") or "." in ticker else [ticker, "^" + ticker]
    last_err = None
    for t in candidates:
        try:
            raw = yf.download(t, start=start, end=end + timedelta(days=1),
                              auto_adjust=False, progress=False)
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.reset_index()
            return _clean_ohlc(raw)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ValueError(f"Yahoo Finance не вернул данных для {ticker}. {last_err or ''}")


def load_uploaded(file, name: str) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
    else:
        raw = file.read()
        text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
        sep = ";" if text.count(";") > text.count(",") else ","
        df = pd.read_csv(io.StringIO(text), sep=sep)
    return _clean_ohlc(df)


def fetch_series(ticker: str, source: str, start: date, end: date) -> pd.DataFrame:
    """source: 'wsj', 'yahoo' или 'auto' (сначала WSJ, затем Yahoo)."""
    errors = []
    order = {"wsj": [fetch_wsj], "yahoo": [fetch_yahoo], "auto": [fetch_wsj, fetch_yahoo]}[source]
    for fn in order:
        try:
            df = fetch_series_inner(fn, ticker, start, end)
            df.attrs["source"] = "WSJ" if fn is fetch_wsj else "Yahoo Finance"
            return df
        except Exception as e:  # noqa: BLE001
            errors.append(f"{'WSJ' if fn is fetch_wsj else 'Yahoo'}: {e}")
    raise ValueError(f"{ticker}: " + " | ".join(errors))


def fetch_series_inner(fn, ticker, start, end):
    df = fn(ticker, start, end)
    df = df.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if len(df) < 10:
        raise ValueError("слишком мало наблюдений")
    return df


# =============================================================================
# 2. ОБЛАКО ТОЧЕК
# =============================================================================
def build_point_cloud(prices: pd.DataFrame, embed_dim: int = 4, delay: int = 1):
    """prices: колонки = индексы (inner join по датам).
    >= 2 индексов: точка дня t = вектор лог-доходностей всех индексов (как в вашем коде).
    1 индекс: вложение Такенса (r_t, r_{t-τ}, ..., r_{t-(d-1)τ}),
    иначе облако одномерное и H1 всегда пустое."""
    p = prices.to_numpy(dtype=float)
    r = np.log(p[1:] / p[:-1])
    dates = prices.index[1:]
    ok = np.all(np.isfinite(r), axis=1)
    r, dates = r[ok], dates[ok]
    if r.shape[1] == 1 and embed_dim > 1:
        x = r[:, 0]
        lag = (embed_dim - 1) * delay
        X = np.column_stack([x[lag - j * delay: len(x) - j * delay] for j in range(embed_dim)])
        return X, dates[lag:]
    return r, dates


# =============================================================================
# 3. ДИАГРАММЫ ПЕРСИСТЕНТНОСТИ
# =============================================================================
def _h0_mst(X):
    d = squareform(pdist(X))
    mst = minimum_spanning_tree(d).toarray()
    deaths = np.sort(mst[mst > 0])
    return np.column_stack([np.zeros_like(deaths), deaths])


def diagrams(X, maxdim=1):
    """Возвращает [H0, H1] без бесконечных точек."""
    if _ripser is not None:
        dg = _ripser(X, maxdim=maxdim)["dgms"]
        return [d[np.isfinite(d[:, 1])] for d in dg]
    if maxdim >= 1:
        raise ImportError("Для H1 нужен пакет ripser (pip install ripser).")
    return [_h0_mst(X)]


# =============================================================================
# 4. WASSERSTEIN (та же формула, что persim.wasserstein: L∞ между точками,
#    евклидово расстояние до диагонали (d-b)/√2, p = 1)
# =============================================================================
def wasserstein(d1, d2):
    d1 = d1[np.isfinite(d1[:, 1])] if len(d1) else np.zeros((0, 2))
    d2 = d2[np.isfinite(d2[:, 1])] if len(d2) else np.zeros((0, 2))
    m, n = len(d1), len(d2)
    diag1 = (d1[:, 1] - d1[:, 0]) / np.sqrt(2) if m else np.zeros(0)
    diag2 = (d2[:, 1] - d2[:, 0]) / np.sqrt(2) if n else np.zeros(0)
    if m == 0 or n == 0:
        return float(diag1.sum() + diag2.sum())
    D = np.zeros((m + n, m + n))
    D[:m, :n] = cdist(d1, d2, metric="chebyshev")
    ur = np.full((m, m), np.inf); np.fill_diagonal(ur, diag1)
    ll = np.full((n, n), np.inf); np.fill_diagonal(ll, diag2)
    D[:m, n:] = ur
    D[m:, :n] = ll
    rows, cols = linear_sum_assignment(D)
    return float(D[rows, cols].sum())


# =============================================================================
# 5. НОРМЫ PERSISTENCE LANDSCAPE (точно, по определению Бубеника:
#    ||λ||_p = (Σ_k ∫ λ_k(t)^p dt)^(1/p) — то же, что PersLandscapeExact.p_norm)
# =============================================================================
def landscape_norms(dgm):
    """Возвращает (L1, L2). Пустая диаграмма -> (0, 0)."""
    if len(dgm) == 0:
        return 0.0, 0.0
    b, d = dgm[:, 0], dgm[:, 1]
    # Все изломы λ_k лежат в точках (x+y)/2, x,y ∈ {b_i} ∪ {d_j}
    ends = np.unique(np.concatenate([b, d]))
    t = np.unique((ends[:, None] + ends[None, :]).ravel() / 2.0)
    t = t[(t >= b.min()) & (t <= d.max())]
    tents = np.clip(np.minimum(t[None, :] - b[:, None], d[:, None] - t[None, :]), 0, None)
    lam = -np.sort(-tents, axis=0)            # λ_1 ≥ λ_2 ≥ ... в каждой точке t
    h = np.diff(t)
    y0, y1 = lam[:, :-1], lam[:, 1:]
    l1 = float(np.sum(h * (y0 + y1) / 2))
    l2sq = float(np.sum(h * (y0 ** 2 + y0 * y1 + y1 ** 2) / 3))
    return l1, float(np.sqrt(l2sq))


# =============================================================================
# 6. ГЛАВНЫЙ РАСЧЁТ
# =============================================================================
def compute_all(X, dates, window=50, homs=(0, 1), progress=None):
    """Скользящее окно длины `window`.
    Нормы: дата = последний день окна.
    Wasserstein: окно [i, i+w) против [i+w, i+2w), дата = последний день второго окна."""
    n_win = len(X) - window + 1
    if n_win <= window:
        raise ValueError(f"Нужно больше данных: {len(X)} точек при окне {window}.")
    maxdim = max(homs)
    dg_all = []
    norms = {f"H{k}_{p}": np.empty(n_win) for k in homs for p in ("L1", "L2")}
    for i in range(n_win):
        dg = diagrams(X[i:i + window], maxdim=maxdim)
        dg_all.append(dg)
        for k in homs:
            l1, l2 = landscape_norms(dg[k])
            norms[f"H{k}_L1"][i] = l1
            norms[f"H{k}_L2"][i] = l2
        if progress and i % 25 == 0:
            progress(i / n_win)
    norms_df = pd.DataFrame(norms, index=dates[window - 1:])

    n_wd = n_win - window
    wd = {f"H{k}_WD": np.array([wasserstein(dg_all[i][k], dg_all[i + window][k])
                                for i in range(n_wd)]) for k in homs}
    wd_df = pd.DataFrame(wd, index=dates[2 * window - 1:])
    if progress:
        progress(1.0)
    return wd_df, norms_df


def thresholds(series: pd.Series, ks=(1, 2, 3, 4), mode="full"):
    """mode='full'  — μ, σ по всей выборке (как в вашей статье);
    mode='expanding' — μ, σ только по прошлым данным (без заглядывания вперёд)."""
    s = series.astype(float)
    if mode == "expanding":
        mu = s.expanding(min_periods=60).mean().shift(1)
        sd = s.expanding(min_periods=60).std().shift(1)
        return {k: mu + k * sd for k in ks}
    mu, sd = s.mean(), s.std()
    return {k: pd.Series(mu + k * sd, index=s.index) for k in ks}


def episodes(series: pd.Series, thr: pd.Series):
    """Непрерывные периоды, когда ряд выше порога: список (начало, конец, дней, пик)."""
    above = (series > thr).fillna(False).to_numpy()
    out, start = [], None
    idx = series.index
    for i, a in enumerate(above):
        if a and start is None:
            start = i
        if (not a or i == len(above) - 1) and start is not None:
            end = i if a else i - 1
            seg = series.iloc[start:end + 1]
            out.append((idx[start], idx[end], end - start + 1, float(seg.max())))
            start = None
    return out
