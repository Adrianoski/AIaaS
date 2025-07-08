# Usa un'immagine base leggera con Python.
# python:3.9-slim-buster è una buona scelta per le applicazioni basate su Debian.
FROM python:3.10-slim-buster

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Copia il requirements.txt e installa le dipendenze
# È buona pratica copiare solo i requirements e installarli prima di copiare tutto il codice,
# in modo da sfruttare il caching dei layer di Docker.
COPY requirements.txt .
COPY NetworkImprover ./NetworkImprover

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e ./NetworkImprover
# Copia lo script di inferenza. Questo script sarà il punto di ingresso per eseguire il modello.
COPY inference_script.py .
COPY prepare_dataset.py .
COPY controls.py .
COPY data ./data

# Il comando predefinito quando il container viene avviato.
# Lo script riceverà il percorso del modello, il percorso dell'input e il percorso dell'output
# come argomenti da riga di comando.
# Il modello e i dati verranno montati come volumi dal server host.
ENTRYPOINT ["python", "-u", "inference_script.py"]
