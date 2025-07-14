import requests
import time
import json

SERVER_URL = 'http://localhost:5000'

def upload_model(file_path, model_name="MyModel", dataset="CIFAR10",
                 lower_metrics=None, higher_metrics=None):
    with open(file_path, 'rb') as f:
        files = {'network_file': f}
        data = {
            'model_name': model_name,
            'dataset': dataset,
            'lower_metrics': json.dumps(lower_metrics or {}),
            'higher_metrics': json.dumps(higher_metrics or {})
        }
        print("Uploading model...")
        response = requests.post(f'{SERVER_URL}/upload-neural-network', files=files, data=data)

    if response.status_code == 200:
        network_id = response.json()['network_id']
        print(f"Model uploaded successfully. Network ID: {network_id}")
        return network_id
    else:
        raise Exception(f"Upload failed: {response.text}")

def run_model(network_id, dataset="CIFAR10", lower_metrics=None, higher_metrics=None):
    print(f"Requesting execution for network {network_id}...")
    payload = {
        "dataset": dataset,
        "metrics": {
            "lower": lower_metrics or {},
            "higher": higher_metrics or {}
        }
    }
    response = requests.post(f'{SERVER_URL}/run-neural-network/{network_id}', json=payload)
    if response.status_code == 202:
        print("Execution started.")
    else:
        raise Exception(f"Execution start failed: {response.text}")

def poll_result(network_id, interval=5):
    print("Polling for results...")
    while True:
        response = requests.get(f'{SERVER_URL}/execution-result/{network_id}')
        result = response.json()

        if result["status"] == "completed":
            print("Execution completed:")
            print(json.dumps(result["results"], indent=2))
            break
        elif result["status"] == "execution_failed":
            print("Execution failed:")
            print(result.get("error", "Unknown error"))
            break
        else:
            print(f"Still running... (status: {result['status']})")
            time.sleep(interval)

if __name__ == "__main__":
    # === CONFIGURATION ===
    model_file = "AutoEncoder.pt"  
    model_name = "AutoEncoder"
    dataset = "esa_anomaly"
    lower_metrics = {"loss": 0.05}
    higher_metrics = {}

    try:
        network_id = upload_model(model_file, model_name, dataset, lower_metrics, higher_metrics)
        run_model(network_id, dataset, lower_metrics, higher_metrics)
        poll_result(network_id)
    except Exception as e:
        print("Error:", str(e))
