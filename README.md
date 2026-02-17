# EasyMag • Warehouse Performance Dashboard (Streamlit)

Dashboard Streamlit per accorpare **5 export Excel EasyMag** (uno per operazione) e visualizzare KPI + grafici
(istogrammi e torte) delle performance dei magazzinieri.

## Cosa fa
- Carica più file Excel (giornalieri o mensili).
- Trasforma le tabelle pivot in formato “long” (data, operatore, quantità, operazione).
- **Accorpa i codici operatore ignorando maiuscole/minuscole**.
- Mappa le operazioni nei 3 reparti:
  - **Sell-In**
  - **Picking**
  - **Controllo & Packaging**
- Dashboard con filtri (date, reparto, operazione) + export Excel.

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

## Come usare
1. Carica i 5 file Excel (uno per ciascuna operazione)
2. Nella sidebar assegna **file → operazione**
3. (Opzionale) modifica **operazione → reparto**
4. Usa filtri e scarica l’Excel con i dati puliti/aggregati
