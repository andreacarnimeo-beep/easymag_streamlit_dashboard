# EasyMag • Warehouse Performance Dashboard (Streamlit) — v2

Dashboard Streamlit per accorpare **5 export Excel EasyMag** e visualizzare KPI + grafici
(istogrammi e torte) delle performance dei magazzinieri.

## Novità v2 (rispetto alla versione precedente)
- **Riconoscimento automatico dell’operazione** dal testo nel file:
  - Riga tipo: `(*) Numero di Operazioni Identificazione da Web.`
- Regole reparto:
  - Picking → solo **Prelievi**
  - Sell‑In → **Identificazioni Web**, **Identificazioni Dirette**, **Depositi RF e Dirette**
    - **Performance reparto Sell‑In** conteggia **solo Depositi RF e Dirette**
  - Controllo & Packaging → solo **Chiusura Colli Web**
- Performance operatori: **sempre tutte e 5 le operazioni**

## Struttura progetto
- `app.py` → app Streamlit
- `requirements.txt` → dipendenze

## Avvio locale
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

## Deploy gratuito su Streamlit Community Cloud
1. Carica questa repo su GitHub
2. Vai su Streamlit Community Cloud → “New app”
3. Seleziona la repo, branch e `app.py`
4. Deploy
