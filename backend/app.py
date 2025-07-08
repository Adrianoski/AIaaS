import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import onnx
import uuid
import json # Nuovo import
import subprocess # Nuovo import per eseguire comandi di sistema
import time # Per gestire file temporanei con nomi unici

from flask import send_from_directory
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploaded_models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# Cartella per i dati temporanei di input/output per Docker
TEMP_DATA_FOLDER = 'temp_docker_data'
os.makedirs(TEMP_DATA_FOLDER, exist_ok=True)

neural_networks_db = {}
OUTPUT_FOLDER = "optimized_models"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/optimized_models/<filename>')
def serve_optimized_model(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    optimized_dir = os.path.join(base_dir, "optimized_models")
    return send_from_directory(optimized_dir, filename, as_attachment=True)


def inspect_pytorch_model(model_path):
    try:
        model = torch.jit.load(model_path)
        model.eval()
        details = {
            "framework": "PyTorch",
            "model_type": "TorchScript",
            "is_cuda_available_on_server": torch.cuda.is_available(),
            "example_input_shape": "Requires model specific logic to infer" # Difficile senza input dummy
        }
        return details
    except Exception as e:
        raise ValueError(f"Errore nell'ispezione del modello PyTorch: {e}")


@app.route('/upload-neural-network', methods=['POST'])
def upload_neural_network():
    if 'network_file' not in request.files:
        return jsonify({"error": "Nessun file 'network_file' nella richiesta"}), 400

    file = request.files['network_file']
    model_name = request.form.get('model_name', 'Modello Sconosciuto')

    if file.filename == '':
        return jsonify({"error": "Nessun file selezionato"}), 400

    if file:
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ['.pt', '.pth', '.onnx']:
            return jsonify({"error": "Formato file non supportato. Usa .pt, .pth o .onnx"}), 400

        network_id = str(uuid.uuid4())
        filename = f"{network_id}{file_extension}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        neural_networks_db[network_id] = {
            "name": model_name,
            "original_filename": file.filename,
            "path": filepath,
            "status": "pending_inspection",
            "details": None,
            "error": None
        }

        try:
            if file_extension in ['.pt', '.pth']:
                details = inspect_pytorch_model(filepath)
            else:
                raise ValueError("Formato file non supportato (dovrebbe essere già stato controllato).")

            neural_networks_db[network_id]["details"] = details
            neural_networks_db[network_id]["status"] = "ready"
            app.logger.info(f"Rete {network_id} ispezionata con successo. Dettagli: {details}")

        except ValueError as e:
            neural_networks_db[network_id]["status"] = "inspection_failed"
            neural_networks_db[network_id]["error"] = str(e)
            app.logger.error(f"Errore durante l'ispezione della rete {network_id}: {e}")
        except Exception as e:
            neural_networks_db[network_id]["status"] = "inspection_failed"
            neural_networks_db[network_id]["error"] = f"Errore generico durante l'ispezione: {e}"
            app.logger.error(f"Errore inatteso durante l'ispezione della rete {network_id}: {e}")

        return jsonify({
            "message": "File caricato e ispezione avviata",
            "network_id": network_id,
            "status": neural_networks_db[network_id]["status"]
        }), 200

@app.route('/neural-networks', methods=['GET'])
def get_neural_networks():
    networks_list = []
    for net_id, net_data in neural_networks_db.items():
        networks_list.append({
            "id": net_id,
            "name": net_data["name"],
            "original_filename": net_data["original_filename"],
            "status": net_data["status"],
            "details": net_data["details"],
            "error": net_data["error"]
        })
    return jsonify(networks_list), 200


@app.route('/run-neural-network/<network_id>', methods=['POST'])
def run_neural_network(network_id):
    if network_id not in neural_networks_db:
        return jsonify({"error": "Rete neurale non trovata"}), 404

    network_data = neural_networks_db[network_id]

    if network_data["status"] != "ready":
        return jsonify({"error": f"La rete non è pronta per l'esecuzione. Stato attuale: {network_data['status']}"}), 400

    # Recupera i dati di input dal body della richiesta JSON
    try:
        # input_data = request.json.get('input_data')
        # if input_data is None:
        #     return jsonify({"error": "Dati di input ('input_data') mancanti nel corpo della richiesta JSON."}), 400
        data = request.get_json()

        metrics = data.get('metrics', {})
        dataset = data.get('dataset', None)

        if not dataset:
            return jsonify({"error": "Dataset non specificato"}), 400

        input_data = {
            "dataset": dataset,
            "metrics": metrics
        }


        # Genera nomi file unici per input e output nel TEMP_DATA_FOLDER
        # Questo garantisce che esecuzioni multiple non si sovrappongano
        timestamp = int(time.time() * 1000) # Millisecondi per maggiore unicità
        unique_id = str(uuid.uuid4())[:8] # Un UUID corto
        input_filename = f"input_{network_id}_{timestamp}_{unique_id}.json"
        output_filename = f"output_{network_id}_{timestamp}_{unique_id}.json"

        input_filepath_host = os.path.join(TEMP_DATA_FOLDER, input_filename)
        print(output_filename)
        output_filepath_host = os.path.join(TEMP_DATA_FOLDER, output_filename)

        # Scrivi i dati di input nel file JSON temporaneo
        with open(input_filepath_host, 'w') as f:
            json.dump({"data": input_data}, f) # Incapsula i dati in una chiave 'data' per lo script di inferenza
        os.makedirs(os.path.dirname(output_filepath_host), exist_ok=True)
        if not os.path.exists(output_filepath_host):
            with open(output_filepath_host, 'w') as f:
                pass
        # Aggiorna lo stato nel DB (prima di avviare il container)
        neural_networks_db[network_id]["status"] = "running"
        neural_networks_db[network_id]["current_execution_output_path"] = output_filepath_host # Salva il path per il polling o recupero

        # Costruisci il comando Docker run
        # -v: monta i volumi. Percorso host:Percorso container
        #     Il modello viene montato in sola lettura.
        #     I file di input/output vengono montati per la comunicazione.
        # --rm: rimuove automaticamente il container una volta terminato.
        # --name: assegna un nome al container (utile per debug).
        # --network none: isola il container dalla rete (ulteriore sicurezza, se non ha bisogno di internet)
        # --memory: limita la memoria (es: 512m)
        # --cpus: limita l'uso della CPU (es: 0.5)
        
        # NOTE: Assicurati che l'immagine Docker 'neural-inference-image' sia stata costruita!
        docker_image_name = 'neural-inference-image'
        
        # Percorsi all'interno del container
        model_path_in_container = f"/app/model/{os.path.basename(network_data['path'])}"
        input_path_in_container = f"/app/data/{input_filename}"
        output_path_in_container = f"/app/data/{output_filename}"
        print('---------------------------------------------------')
        print(output_filepath_host)
        print('---------------------------------------------------')
        docker_command = [
            'docker', 'run',
            '--rm',
            f'--name=inference_{network_id}_{timestamp}', # Nome univoco per il container
            
            '-v', f'{os.path.abspath(network_data["path"])}:{model_path_in_container}:ro', # Modello in sola lettura
            '-v', f'{os.path.abspath(input_filepath_host)}:{input_path_in_container}', # Input
            '-v', f'{os.path.abspath(output_filepath_host)}:{output_path_in_container}', # Output
            docker_image_name,
            model_path_in_container,
            input_path_in_container,
            output_path_in_container
        ]

        app.logger.info(f"Avvio comando Docker: {' '.join(docker_command)}")

        # Esegui il comando Docker in background (senza bloccare il server Flask)
        # In un sistema più robusto, useresti una coda di lavoro (Celery) per questo.
        # Per ora, usiamo subprocess.Popen per non bloccare la risposta al client.
        # stderr e stdout vengono reindirizzati per la diagnostica.
        #process = subprocess.Popen(docker_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # DEBUG: esegue Docker e stampa subito l'output del container
        process = subprocess.Popen(docker_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, _ = process.communicate()
        print("------ OUTPUT DOCKER START ------")
        print(stdout.decode())
        print("------ OUTPUT DOCKER END   ------") 

        
        # Non aspettiamo che il processo finisca qui.
        # L'aggiornamento dello stato e la lettura dei risultati avverranno tramite un meccanismo
        # separato (es. polling da parte del client o un worker in background).
        
        # Potresti salvare il PID del processo o il nome del container per il monitoraggio
        neural_networks_db[network_id]["docker_process_pid"] = process.pid
        neural_networks_db[network_id]["docker_container_name"] = f'inference_{network_id}_{timestamp}'

        return jsonify({
            "message": "Esecuzione della rete avviata in un container Docker.",
            "network_id": network_id,
            "status": "running",
            "output_expected_at": output_filepath_host # Indica dove il risultato sarà disponibile
        }), 202 # Accepted (l'elaborazione è iniziata, ma non è completata)

    except Exception as e:
        # In caso di errore prima di avviare il container
        neural_networks_db[network_id]["status"] = "execution_failed"
        neural_networks_db[network_id]["error"] = str(e)
        app.logger.error(f"Errore durante l'avvio dell'esecuzione della rete {network_id}: {e}")
        return jsonify({"error": f"Errore durante l'avvio dell'esecuzione: {e}"}), 500

# Nuovo endpoint per recuperare i risultati di una specifica esecuzione
@app.route('/execution-result/<network_id>', methods=['GET'])
def get_execution_result(network_id):
    if network_id not in neural_networks_db:
        return jsonify({"error": "Rete neurale non trovata"}), 404

    network_data = neural_networks_db[network_id]
    output_path = network_data.get("current_execution_output_path")

    if not output_path or not os.path.exists(output_path):
        return jsonify({"message": "Risultati non ancora disponibili o esecuzione non avviata.", "status": network_data["status"]}), 200

    try:
        with open(output_path, 'r') as f:
            results = json.load(f)
        
        # Aggiorna lo stato se il risultato è stato trovato e non è un errore
        if "error" not in results:
            neural_networks_db[network_id]["status"] = "completed"
        else:
            neural_networks_db[network_id]["status"] = "execution_failed"
            neural_networks_db[network_id]["error"] = results.get("error", "Errore sconosciuto nel container")

        # Rimuovi i file temporanei dopo averli letti (opzionale, ma buona pratica)
        # os.remove(output_path)
        # if os.path.exists(output_path.replace('output_', 'input_')): # Trova il file input corrispondente
        #     os.remove(output_path.replace('output_', 'input_'))
        
        return jsonify({
            "network_id": network_id,
            "status": neural_networks_db[network_id]["status"],
            "results": results # Contiene l'output del modello o il messaggio di errore
        }), 200
    except json.JSONDecodeError:
        neural_networks_db[network_id]["status"] = "execution_failed"
        neural_networks_db[network_id]["error"] = "Il file di output non è un JSON valido."
        return jsonify({"error": "Impossibile leggere i risultati: file di output corrotto.", "status": "execution_failed"}), 500
    except Exception as e:
        neural_networks_db[network_id]["status"] = "execution_failed"
        neural_networks_db[network_id]["error"] = f"Errore generico nel recupero risultati: {e}"
        return jsonify({"error": f"Errore durante il recupero dei risultati: {e}", "status": "execution_failed"}), 500


# --- Esecuzione dell'Applicazione ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)