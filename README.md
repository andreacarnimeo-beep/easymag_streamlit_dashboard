# EasyMag • Warehouse Performance Dashboard (Streamlit) — v3

## Fix: rimozione duplicazione tabella
In alcuni export EasyMag la pivot viene duplicata: compare un secondo header `Operatori/Data` più sotto.

✅ Questa versione usa **solo la prima tabella**, tagliando tutto ciò che segue la seconda occorrenza di `Operatori/Data`.
