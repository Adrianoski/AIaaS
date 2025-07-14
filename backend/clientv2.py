import threading
import time
import requests
import json

SERVER_URL = 'http://localhost:5000'

def fake_sum_task():
    for i in range(120):
        a = i + 100
        print(f"[SUM] {i} + 100 = {a}")
        time.sleep(1)

def fake_product_task():
    for i in range(120):
        a = i * 7
        print(f"[PRODUCT] {i} * 7 = {a}")
        time.sleep(1.2)

def upload_and_poll_model(file_path, model_name="ThreadedModel", dataset="esa_anomaly"):
    lower_metrics = {"loss": 0.05}
    higher_metrics = {}

    try:
        # Upload
        with open(file_path, 'rb') as f:
            files = {'network_file': f}
            data = {
                'model_name': model_name,
                'dataset': dataset,
                'lower_metrics': json.dumps(lower_metrics),
                'higher_metrics': json.dumps(higher_metrics)
            }
            print("[SERVER] Uploading model...")
            response = requests.post(f'{SERVER_URL}/upload-neural-network', files=files, data=data)

        if response.status_code != 200:
            print(f"[SERVER] Upload failed: {response.text}")
            return

        network_id = response.json()['network_id']
        print(f"[SERVER] Model uploaded. Network ID: {network_id}")

        # Run
        payload = {
            "dataset": dataset,
            "metrics": {
                "lower": lower_metrics,
                "higher": higher_metrics
            }
        }
        response = requests.post(f'{SERVER_URL}/run-neural-network/{network_id}', json=payload)
        if response.status_code != 202:
            print(f"[SERVER] Run failed: {response.text}")
            return
        print("[SERVER] Execution started.")

        # Poll
        print("[SERVER] Polling for result...")
        while True:
            response = requests.get(f'{SERVER_URL}/execution-result/{network_id}')
            result = response.json()
            status = result.get("status")

            if status == "completed":
                print("[SERVER] Execution completed:")
                print(json.dumps(result["results"], indent=2))
                break
            elif status == "execution_failed":
                print("[SERVER] Execution failed:", result.get("error", "Unknown error"))
                break
            else:
                print("[SERVER] Still running...")
                time.sleep(5)

    except Exception as e:
        print("[SERVER] Error:", str(e))


if __name__ == "__main__":
    model_path = "AutoEncoder.pt"  # Path to a real model file

    # Thread 1: somma
    t1 = threading.Thread(target=fake_sum_task)

    # Thread 2: prodotto
    t2 = threading.Thread(target=fake_product_task)

    # Thread 3: comunicazione con server
    t3 = threading.Thread(target=upload_and_poll_model, args=(model_path,))

    # Avvio dei thread
    t1.start()
    t2.start()
    t3.start()

    # Attendo la fine di tutti i thread
    t1.join()
    t2.join()
    t3.join()

    print("All tasks completed.")
