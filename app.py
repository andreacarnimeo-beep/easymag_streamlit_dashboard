import io
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="EasyMag • Warehouse Performance Dashboard", layout="wide")

OPERATIONS = [
    "Prelievi",
    "Identificazioni Web",
    "Identificazioni Dirette",
    "Controlli Web",
    "Chiusura Colli Web",
]

DEFAULT_DEPT_MAP = {
    "Prelievi": "Picking",
    "Identificazioni Web": "Sell-In",
    "Identificazioni Dirette": "Sell-In",
    "Controlli Web": "Controllo & Packaging",
    "Chiusura Colli Web": "Controllo & Packaging",
}

DEPARTMENTS = ["Sell-In", "Picking", "Controllo & Packaging"]

def _clean_operator(x: str) -> str:
    if x is None:
        return ""
    return str(x).strip().upper()

def _pick_date_col(df: pd.DataFrame) -> str:
    # The exports usually have "Operatori/Data" but we keep it robust.
    candidates = [c for c in df.columns if str(c).strip().lower() in {"operatori/data", "data", "date", "giorno"}]
    if candidates:
        return candidates[0]
    return df.columns[0]

def _parse_date_series(s: pd.Series) -> pd.Series:
    # Try datetime first, then string parsing, then monthly period parsing.
    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s).dt.normalize()

    s2 = s.astype(str).str.strip()
    # Common EasyMag export format: YYYY-MM-DD
    dt = pd.to_datetime(s2, errors="coerce", dayfirst=False)
    if dt.notna().mean() > 0.8:
        return dt.dt.normalize()

    # Try month formats like "2026-01" or "01/2026" or "Gennaio 2026"
    # 1) YYYY-MM
    dt2 = pd.to_datetime(s2 + "-01", errors="coerce")
    if dt2.notna().mean() > 0.6:
        return dt2.dt.normalize()

    # 2) MM/YYYY
    dt3 = pd.to_datetime("01/" + s2, errors="coerce", dayfirst=True)
    if dt3.notna().mean() > 0.6:
        return dt3.dt.normalize()

    return dt  # may be mostly NaT

@st.cache_data(show_spinner=False)
def load_pivot_excel(file_bytes: bytes, operation: str) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))
    date_col = _pick_date_col(df)

    # Drop total columns
    drop_cols = [c for c in df.columns if str(c).strip().lower() in {"tot:", "tot", "total", "totale"}]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Normalize operator columns to uppercase and merge duplicates (case differences)
    cols = list(df.columns)
    new_cols = []
    for c in cols:
        if c == date_col:
            new_cols.append(c)
        else:
            new_cols.append(_clean_operator(c))
    df.columns = new_cols

    # If date col got uppercased by coincidence (rare), re-find it
    date_col = _pick_date_col(df)

    # Merge duplicate operator columns (e.g., ALOIUDICE and ALoiudice)
    op_cols = [c for c in df.columns if c != date_col]
    if len(op_cols) != len(set(op_cols)):
        merged = df[[date_col]].copy()
        for c in sorted(set(op_cols)):
            merged[c] = pd.to_numeric(df.loc[:, df.columns == c].sum(axis=1), errors="coerce").fillna(0)
        df = merged
        op_cols = [c for c in df.columns if c != date_col]

    df[date_col] = _parse_date_series(df[date_col])
    df = df.dropna(subset=[date_col]).copy()

    long_df = df.melt(id_vars=[date_col], value_vars=op_cols, var_name="operatore", value_name="qta")
    long_df["operatore"] = long_df["operatore"].map(_clean_operator)
    long_df["qta"] = pd.to_numeric(long_df["qta"], errors="coerce").fillna(0).astype(float)

    long_df = long_df[long_df["qta"] > 0].copy()
    long_df = long_df.rename(columns={date_col: "data"})
    long_df["operazione"] = operation
    long_df["mese"] = long_df["data"].dt.to_period("M").astype(str)
    return long_df

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

st.title("📦 EasyMag • Performance magazzinieri")

with st.sidebar:
    st.header("1) Carica file")
    uploaded = st.file_uploader(
        "Carica 5 export Excel (uno per operazione).",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
    )

    st.divider()
    st.header("2) Mappa file → operazione")
    st.caption("Se l'app non riesce a capire automaticamente il tipo di operazione, assegnalo qui.")
    file_ops = {}
    if uploaded:
        for f in uploaded:
            key = f"{f.name}__op"
            file_ops[f.name] = st.selectbox(
                f"**{f.name}**",
                OPERATIONS,
                index=min(len(file_ops), len(OPERATIONS) - 1),
                key=key,
            )

    st.divider()
    st.header("3) Mappa operazione → reparto")
    dept_map = {}
    for op in OPERATIONS:
        dept_map[op] = st.selectbox(op, DEPARTMENTS, index=DEPARTMENTS.index(DEFAULT_DEPT_MAP[op]), key=f"dept_{op}")

    st.divider()
    st.header("Filtri")
    agg_level = st.radio("Vista dati", ["Giornaliera", "Mensile"], horizontal=True)

data = pd.DataFrame()
if uploaded:
    frames = []
    for f in uploaded:
        op = file_ops.get(f.name, OPERATIONS[0])
        frames.append(load_pivot_excel(f.getvalue(), op))
    if frames:
        data = pd.concat(frames, ignore_index=True)

if data.empty:
    st.info("Carica almeno un file Excel per iniziare.")
    st.stop()

data["reparto"] = data["operazione"].map(dept_map)

# Filters after loading
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

# Aggregations
if agg_level == "Mensile":
    view = (
        filtered.groupby(["mese", "reparto", "operazione", "operatore"], as_index=False)["qta"].sum()
        .rename(columns={"mese": "periodo"})
    )
else:
    view = (
        filtered.groupby(["data", "reparto", "operazione", "operatore"], as_index=False)["qta"].sum()
        .rename(columns={"data": "periodo"})
    )

# KPI row
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
    st.subheader("Distribuzione per reparto e per operazione")
    c1, c2 = st.columns(2)
    rep = filtered.groupby("reparto", as_index=False)["qta"].sum().sort_values("qta", ascending=False)
    op = filtered.groupby("operazione", as_index=False)["qta"].sum().sort_values("qta", ascending=False)

    with c1:
        fig = px.pie(rep, names="reparto", values="qta", title="Quota per reparto")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(op, names="operazione", values="qta", title="Quota per operazione")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 15 operatori (nel filtro)")
    top = filtered.groupby("operatore", as_index=False)["qta"].sum().sort_values("qta", ascending=False).head(15)
    fig = px.bar(top, x="operatore", y="qta", title="Top 15 operatori • Totale operazioni")
    st.plotly_chart(fig, use_container_width=True)

with tab_ops:
    st.subheader("Classifica operatori e breakdown")
    left, right = st.columns([1.2, 1])
    with left:
        by_op = filtered.groupby(["operatore", "reparto"], as_index=False)["qta"].sum()
        piv = by_op.pivot_table(index="operatore", columns="reparto", values="qta", aggfunc="sum", fill_value=0)
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

        by_op_rep = df_op.groupby("reparto", as_index=False)["qta"].sum().sort_values("qta", ascending=False)
        fig = px.pie(by_op_rep, names="reparto", values="qta", title=f"{sel_op} • Quota per reparto")
        st.plotly_chart(fig, use_container_width=True)

with tab_rep:
    st.subheader("Reparti: ranking e composizione")
    rep_rank = filtered.groupby(["reparto", "operatore"], as_index=False)["qta"].sum()
    rep_total = filtered.groupby("reparto", as_index=False)["qta"].sum().sort_values("qta", ascending=False)

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(rep_total, x="reparto", y="qta", title="Totale per reparto")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sel_rep = st.selectbox("Seleziona reparto", DEPARTMENTS, index=0)
        top_rep = (
            rep_rank[rep_rank["reparto"] == sel_rep]
            .sort_values("qta", ascending=False)
            .head(15)
        )
        fig = px.bar(top_rep, x="operatore", y="qta", title=f"Top 15 operatori • {sel_rep}")
        st.plotly_chart(fig, use_container_width=True)

with tab_trend:
    st.subheader("Trend nel tempo")
    # Total over time
    if agg_level == "Mensile":
        t = filtered.groupby(["mese"], as_index=False)["qta"].sum().rename(columns={"mese": "periodo"})
    else:
        t = filtered.groupby(["data"], as_index=False)["qta"].sum().rename(columns={"data": "periodo"})
    fig = px.line(t, x="periodo", y="qta", title="Totale operazioni nel tempo", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trend per reparto")
    if agg_level == "Mensile":
        tr = filtered.groupby(["mese", "reparto"], as_index=False)["qta"].sum().rename(columns={"mese": "periodo"})
    else:
        tr = filtered.groupby(["data", "reparto"], as_index=False)["qta"].sum().rename(columns={"data": "periodo"})
    fig = px.line(tr, x="periodo", y="qta", color="reparto", title="Reparti nel tempo", markers=True)
    st.plotly_chart(fig, use_container_width=True)

with tab_export:
    st.subheader("Scarica i dati (puliti e aggregati)")
    st.caption("Include: dati long (puliti), vista giornaliera, vista mensile.")
    daily = (
        data.groupby(["data", "reparto", "operazione", "operatore"], as_index=False)["qta"].sum()
        .sort_values(["data", "reparto", "operazione", "operatore"])
    )
    monthly = (
        data.groupby(["mese", "reparto", "operazione", "operatore"], as_index=False)["qta"].sum()
        .rename(columns={"mese": "mese"})
        .sort_values(["mese", "reparto", "operazione", "operatore"])
    )
    excel_bytes = to_excel_bytes({
        "raw_long": data.sort_values(["data", "operatore", "operazione"]),
        "daily": daily,
        "monthly": monthly,
    })
    st.download_button(
        "⬇️ Download Excel",
        data=excel_bytes,
        file_name=f"easymag_dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Note su codici operatore")
    st.write(
        "I codici operatore vengono accorpati ignorando maiuscole/minuscole "
        "(es. `ALoiudice` e `ALOIUDICE` → `ALOIUDICE`). "
        "Se nello stesso file ci sono colonne duplicate, vengono sommate automaticamente."
    )
