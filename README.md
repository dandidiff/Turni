# Analisi Turnazioni vs Vendite

Questo strumento analizza la relazione tra i dati di vendita dei negozi e le turnazioni del personale, identificando eventuali anomalie nella distribuzione del personale rispetto alle vendite storiche.

## Requisiti

- Python 3.8 o superiore
- pip (gestore pacchetti Python)

## Installazione

1. Clona questo repository
2. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

## Utilizzo

1. Avvia l'applicazione:
```bash
streamlit run app.py
```

2. L'applicazione si aprirà nel tuo browser predefinito

3. Carica il file CSV delle turnazioni con il seguente formato:
   - data (YYYY-MM-DD)
   - num_persone (numero di persone programmate)

4. Clicca su "Analizza Turnazioni" per vedere i risultati

## Output

L'applicazione mostrerà:
- Una tabella con le anomalie rilevate
- Un grafico che visualizza le vendite per persona per ogni giorno della settimana
- I giorni in cui il numero di persone programmate non è proporzionale alle vendite storiche

## Note

- I dati di vendita sono attualmente di esempio. In un ambiente di produzione, questi dovrebbero essere sostituiti con dati reali.
- Le anomalie vengono calcolate utilizzando la deviazione standard per identificare giorni con un rapporto vendite/personale significativamente diverso dalla media. 