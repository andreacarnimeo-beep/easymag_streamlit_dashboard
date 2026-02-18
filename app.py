import io
import re
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="EasyMag • Warehouse Performance Dashboard", layout="wide")

# ---- Business rules (as requested) ----
# Files/operations expected (5):
# - Picking: Prelievi
# - Sell-In: Identificazioni Web, Identificazioni Dirette, Depositi RF e Dirette
#   -> "Performance reparto Sell-In" uses ONLY "Depositi RF e Dirette"
# - Controllo & Packaging: Chiusura Colli Web
#
# "Performance operatori" uses ALL 5 operations.

OPERATIONS = [
    "Prelievi",
    "Identificazioni Web",
    "Identificazioni Dirette",
    "Depositi RF e Dirette",
    "Chiusura Colli Web",
]

DEFAULT_DEPT_MAP = {
    "Prelievi": "Picking",
    "Identificazioni Web": "Sell-In",
    "Identificazioni Dirette": "Sell-In",
    "Depositi RF e Dirette": "Sell-In",
    "Chiusura Colli Web": "Controllo & Packaging",
}

DEPARTMENTS = ["Sell-In", "Picking", "Controllo & Packaging"]

# ---- Parsing helpers ----
def _clean_operator(x: str) -> str:
    if x is None:
        return ""
    return str(x).strip().upper()

def _find_header_row(raw: pd.DataFrame) -> int:
    """Find the pivot header row where first column is 'Operatori/Data' (case-insensitive)."""
    for i in range(min(raw.shape[0], 500)):
        v = raw.iat[i, 0]
        if isinstance(v, str) and v.strip().lower() == "operatori/data":
            return i
    return 0  # fallback

def _detect_operation(raw: pd.DataFrame) -> str | None:
    """
    Detect operation from a row like: '(*) Numero di Operazioni Identificazione da Web.'
    We'll scan the first column for that signature.
    """
    op_text = None
    for i in range(min(raw.shape[0], 800)):
        v = raw.iat[i, 0]
        if isinstance(v, str) and "numero di operazioni" in v.lower():
            op_text = v.strip().lower()
            break

    if not op_text:
        return None

    # Keyword mapping (robust)
    if "preliev" in op_text:
        return "Prelievi"
    if "identificazione" in op_text and "web" in op_text:
        return "Identificazioni Web"
    # EasyMag sometimes labels direct identifications as "Procedura Diretta."
    if "procedura diretta" in op_text or ("diretta" in op_text and "deposit" not in op_text):
        return "Identificazioni Dirette"
    if "deposit" in op_text:
        return "Depositi RF e Dirette"
    if "chiusura" in op_text or "colli" in op_text:
        return "Chiusura Colli Web"

    return None

def _parse_date_series(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s).dt.normalize()

    s2 = s.astype(str).str.strip()

    # 1) Try standard daily dates (YYYY-MM-DD, DD/MM/YYYY, etc.)
    dt = pd.to_datetime(s2, errors="coerce", dayfirst=True)
    if dt.notna().mean() > 0.8:
        return dt.dt.normalize()

    # 2) Monthly formats like YYYY-MM or MM/YYYY
    dt2 = pd.to_datetime(s2 + "-01", errors="coerce")
    if dt2.notna().mean() > 0.6:
        return dt2.dt.normalize()

    dt3 = pd.to_datetime("01/" + s2, errors="coerce", dayfirst=True)
    if dt3.notna().mean() > 0.6:
        return dt3.dt.normalize()

    return dt

@st.cache_data(show_spinner=False)
def load_easymag_excel(file_bytes: bytes) -> tuple[pd.DataFrame, str | None]:
    """
    Returns:
      - long dataframe with columns: data, mese, operatore, qta
      - detected operation (or None)
    """
    raw = pd.read_excel(io.BytesIO(file_bytes), header=None)
    detected = _detect_operation(raw)

    header_row = _find_header_row(raw)
    df = pd.read_excel(io.BytesIO(file_bytes), header=header_row)

    # Identify date column
    date_candidates = [c for c in df.columns if str(c).strip().lower() in {"operatori/data", "data", "date", "giorno"}]
    date_col = date_candidates[0] if date_candidates else df.columns[0]

    # Drop total columns
    drop_cols = [c for c in df.columns if str(c).strip().lower() in {"tot:", "tot", "total", "totale"}]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Normalize operator columns to uppercase, preserving date col name
    new_cols = []
    for c in df.columns:
        if c == date_col:
            new_cols.append(c)
        else:
            new_cols.append(_clean_operator(c))
    df.columns = new_cols

    # Merge duplicate operator columns (case differences)
    op_cols = [c for c in df.columns if c != date_col]
    if len(op_cols) != len(set(op_cols)):
        merged = df[[date_col]].copy()
        for c in sorted(set(op_cols)):
            merged[c] = pd.to_numeric(df.loc[:, df.columns == c].sum(axis=1), errors="coerce").fillna(0)
        df = merged
        op_cols = [c for c in df.columns if c != date_col]

    # Parse dates (daily or monthly)
    df[date_col] = _parse_date_series(df[date_col])
    df = df.dropna(subset=[date_col]).copy()

    long_df = df.melt(id_vars=[date_col], value_vars=op_cols, var_name="operatore", value_name="qta")
    long_df["operatore"] = long_df["operatore"].map(_clean_operator)
    long_df["qta"] = pd.to_numeric(long_df["qta"], errors="coerce").fillna(0).astype(float)
    long_df = long_df[long_df["qta"] > 0].copy()
    long_df = long_df.rename(columns={date_col: "data"})
    long_df["mese"] = long_df["data"].dt.to_period("M").astype(str)

    return long_df, detected

def compute_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    days = df["data"].nunique()
    total = df["qta"].sum()
    ops = df.groupby("operazione")["qta"].sum().sort_values(ascending=False)
    top_op = ops.index[0] if len(ops) else "-"
    top_operator = df.groupby("operatore")["qta"].sum().sort_values(ascending=False).head(1)
    top_operator_name = top_operator.index[0] if len(top_operator) else "-"
    top_operator_qty = float(top_operator.iloc[0]) if len(top_operator) else 0.0
    return {
        "Totale operazioni": int(round(total)),
        "Giorni (nel filtro)": int(days),
        "Media operazioni/giorno": float(total / max(days, 1)),
        "Operazione principale": top_op,
        "Top operatore": f"{top_operator_name} ({int(round(top_operator_qty))})",
    }

def to_excel_bytes(dfs: dict) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, d in dfs.items():
            d.to_excel(writer, index=False, sheet_name=name[:31])
    return out.getvalue()

def reparto_performance_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reparto KPI rule:
      - Sell-In counts ONLY 'Depositi RF e Dirette'
      - Picking counts Prelievi
      - Controllo & Packaging counts Chiusura Colli Web
    """
    if df.empty:
        return df
    return df[
        (df["reparto"] != "Sell-In") | (df["operazione"] == "Depositi RF e Dirette")
    ].copy()

# ---- UI ----
st.title("📦 EasyMag • Performance magazzinieri")

with st.sidebar:
    st.header("1) Carica file")
    uploaded = st.file_uploader(
        "Carica 5 export Excel EasyMag (uno per operazione).",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
    )

    st.divider()
    st.header("2) Riconoscimento operazione")
    st.caption("L'app legge automaticamente la riga '(*) Numero di Operazioni ...'. Se non riesce, puoi correggere qui.")
    overrides = {}

    detected_map = {}
    if uploaded:
        for f in uploaded:
            _, det = load_easymag_excel(f.getvalue())
            detected_map[f.name] = det

        for f in uploaded:
            det = detected_map.get(f.name)
            label = det if det else "⚠️ Non riconosciuta"
            st.write(f"**{f.name}** → {label}")
            if not det:
                overrides[f.name] = st.selectbox(
                    f"Assegna operazione per {f.name}",
                    OPERATIONS,
                    key=f"ovr_{f.name}",
                )

    st.divider()
    st.header("Filtri")
    agg_level = st.radio("Vista dati", ["Giornaliera", "Mensile"], horizontal=True)

data = pd.DataFrame()

if uploaded:
    frames = []
    for f in uploaded:
        df_long, det = load_easymag_excel(f.getvalue())
        op = det or overrides.get(f.name)
        if not op:
            # Skip if still unknown (should be rare, user can override)
            continue
        df_long["operazione"] = op
        frames.append(df_long)

    if frames:
        data = pd.concat(frames, ignore_index=True)

if data.empty:
    st.info("Carica i file: l'app li unirà e mostrerà la dashboard. Se un file non viene riconosciuto, assegnagli un'operazione nella sidebar.")
    st.stop()

# Apply dept mapping
data["reparto"] = data["operazione"].map(DEFAULT_DEPT_MAP)

# Date bounds
min_d, max_d = data["data"].min().date(), data["data"].max().date()
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
with c1:
    date_from = st.date_input("Dal", value=min_d, min_value=min_d, max_value=max_d)
with c2:
    date_to = st.date_input("Al", value=max_d, min_value=min_d, max_value=max_d)
with c3:
    rep_filter = st.multiselect("Reparto", DEPARTMENTS, default=DEPARTMENTS)
with c4:
    op_filter = st.multiselect("Operazione", OPERATIONS, default=OPERATIONS)

filtered = data[
    (data["data"].dt.date >= date_from)
    & (data["data"].dt.date <= date_to)
    & (data["reparto"].isin(rep_filter))
    & (data["operazione"].isin(op_filter))
].copy()

if filtered.empty:
    st.warning("Nessun dato nel filtro selezionato.")
    st.stop()

# KPI row (operators overall: all 5 operations)
kpis = compute_kpis(filtered)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Totale", f"{kpis.get('Totale operazioni', 0):,}".replace(",", "."))
k2.metric("Giorni", kpis.get("Giorni (nel filtro)", 0))
k3.metric("Media/giorno", f"{kpis.get('Media operazioni/giorno', 0):,.1f}".replace(",", "X").replace(".", ",").replace("X", "."))
k4.metric("Operazione #1", kpis.get("Operazione principale", "-"))
k5.metric("Top operatore", kpis.get("Top operatore", "-"))

tab_over, tab_ops, tab_rep, tab_trend, tab_export = st.tabs(
    ["Panoramica", "Operatori", "Reparti", "Trend", "Export"]
)

with tab_over:
    st.subheader("Panoramica (operatori: tutte le operazioni)")
    c1, c2 = st.columns(2)
    rep_all = filtered.groupby("reparto", as_index=False)["qta"].sum().sort_values("qta", ascending=False)
    op_all = filtered.groupby("operazione", as_index=False)["qta"].sum().sort_values("qta", ascending=False)

    with c1:
        fig = px.pie(rep_all, names="reparto", values="qta", title="Quota per reparto (tutte le operazioni)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(op_all, names="operazione", values="qta", title="Quota per operazione")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Panoramica (performance reparto secondo regola)")
    rep_perf = reparto_performance_view(filtered).groupby("reparto", as_index=False)["qta"].sum().sort_values("qta", ascending=False)
    fig = px.bar(rep_perf, x="reparto", y="qta", title="Performance reparto (Sell‑In = solo Depositi RF e Dirette)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 15 operatori (nel filtro, tutte le operazioni)")
    top = filtered.groupby("operatore", as_index=False)["qta"].sum().sort_values("qta", ascending=False).head(15)
    fig = px.bar(top, x="operatore", y="qta", title="Top 15 operatori • Totale operazioni")
    st.plotly_chart(fig, use_container_width=True)

with tab_ops:
    st.subheader("Performance operatori (tutte le 5 operazioni)")
    left, right = st.columns([1.2, 1])
    with left:
        by_op = filtered.groupby(["operatore", "operazione"], as_index=False)["qta"].sum()
        piv = by_op.pivot_table(index="operatore", columns="operazione", values="qta", aggfunc="sum", fill_value=0)
        piv["Totale"] = piv.sum(axis=1)
        piv = piv.sort_values("Totale", ascending=False)
        st.dataframe(piv, use_container_width=True, height=420)
    with right:
        operator_list = sorted(filtered["operatore"].unique().tolist())
        sel_op = st.selectbox("Dettaglio operatore", operator_list)
        df_op = filtered[filtered["operatore"] == sel_op]
        by_op_op = df_op.groupby("operazione", as_index=False)["qta"].sum().sort_values("qta", ascending=False)
        fig = px.bar(by_op_op, x="operazione", y="qta", title=f"{sel_op} • Operazioni per tipo")
        st.plotly_chart(fig, use_container_width=True)

        by_op_rep_all = df_op.groupby("reparto", as_index=False)["qta"].sum().sort_values("qta", ascending=False)
        fig = px.pie(by_op_rep_all, names="reparto", values="qta", title=f"{sel_op} • Quota per reparto (tutte operazioni)")
        st.plotly_chart(fig, use_container_width=True)

with tab_rep:
    st.subheader("Performance reparti (Sell‑In = solo Depositi RF e Dirette)")
    rep_df = reparto_performance_view(filtered)

    rep_total = rep_df.groupby("reparto", as_index=False)["qta"].sum().sort_values("qta", ascending=False)
    c1, c2 = st.columns([1, 1])

    with c1:
        fig = px.bar(rep_total, x="reparto", y="qta", title="Totale per reparto (regola applicata)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sel_rep = st.selectbox("Seleziona reparto", DEPARTMENTS, index=0)
        rep_rank = rep_df[rep_df["reparto"] == sel_rep].groupby(["operatore"], as_index=False)["qta"].sum()
        top_rep = rep_rank.sort_values("qta", ascending=False).head(15)
        fig = px.bar(top_rep, x="operatore", y="qta", title=f"Top 15 operatori • {sel_rep} (regola reparto)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption("Nota: la tab 'Operatori' mostra sempre tutte le operazioni. Qui la metrica reparto segue la regola richiesta.")

with tab_trend:
    st.subheader("Trend (operatori: tutte le operazioni)")
    if agg_level == "Mensile":
        t = filtered.groupby(["mese"], as_index=False)["qta"].sum().rename(columns={"mese": "periodo"})
    else:
        t = filtered.groupby(["data"], as_index=False)["qta"].sum().rename(columns={"data": "periodo"})
    fig = px.line(t, x="periodo", y="qta", title="Totale operazioni nel tempo", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trend per reparto (regola reparto)")
    rep_df = reparto_performance_view(filtered)
    if agg_level == "Mensile":
        tr = rep_df.groupby(["mese", "reparto"], as_index=False)["qta"].sum().rename(columns={"mese": "periodo"})
    else:
        tr = rep_df.groupby(["data", "reparto"], as_index=False)["qta"].sum().rename(columns={"data": "periodo"})
    fig = px.line(tr, x="periodo", y="qta", color="reparto", title="Reparti nel tempo (regola reparto)", markers=True)
    st.plotly_chart(fig, use_container_width=True)

with tab_export:
    st.subheader("Scarica i dati (puliti e aggregati)")
    st.caption("Include: dati long (puliti), vista giornaliera, vista mensile. (Operatori: tutte le operazioni)")
    daily = (
        data.groupby(["data", "reparto", "operazione", "operatore"], as_index=False)["qta"].sum()
        .sort_values(["data", "reparto", "operazione", "operatore"])
    )
    monthly = (
        data.groupby(["mese", "reparto", "operazione", "operatore"], as_index=False)["qta"].sum()
        .sort_values(["mese", "reparto", "operazione", "operatore"])
    )
    rep_perf = reparto_performance_view(data).groupby(["data", "reparto"], as_index=False)["qta"].sum()

    excel_bytes = to_excel_bytes({
        "raw_long": data.sort_values(["data", "operatore", "operazione"]),
        "daily": daily,
        "monthly": monthly,
        "reparto_perf_daily": rep_perf.sort_values(["data", "reparto"]),
    })
    st.download_button(
        "⬇️ Download Excel",
        data=excel_bytes,
        file_name=f"easymag_dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.divider()
    st.write(
        "✅ Accorpamento operatori: ignora maiuscole/minuscole (match lettera per lettera). "
        "✅ Riconoscimento operazione: usa la riga '(*) Numero di Operazioni ...'. "
        "✅ Performance reparto Sell‑In: solo 'Depositi RF e Dirette' (come richiesto)."
    )
