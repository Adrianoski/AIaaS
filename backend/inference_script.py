import sys
import os
import json
import torch
import numpy as np
from prepare_dataset import prepare_dataset_from_json, train_func, test_func, compute_metrics
from controls import validate_model_against_sdl_for_frontend, recreate_sequential_model_from_jit
from NetworkImprover import Improver as NI
from NetworkImprover import Networks_utils as NU
import time
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torchsummary import summary

layer_constructors = {
    'Linear': nn.Linear,
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'Conv1d': nn.Conv1d,
    'ConvTranspose1d': nn.ConvTranspose1d,
    'BatchNorm1d': nn.BatchNorm1d,
    'Dropout': nn.Dropout,
    'Flatten': nn.Flatten,
    'MaxPool1d': nn.MaxPool1d,
    'AvgPool1d': nn.AvgPool1d,
}
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, create_network=None, layer_dict=None,  hidden_dim=256, latent_dim=64, output_dim=1, layer_constructors=None, window_size=50):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.input_shape = input_dim
        self.output_dim = input_dim * window_size
        self.window_size = window_size
        flattened_dim = input_dim * window_size
        
        if layer_dict is None:
            self.encoder_layers = [
                nn.Sequential(
                    nn.Linear(flattened_dim, hidden_dim),
                    nn.ReLU()),
                nn.Sequential(
                    nn.Linear(hidden_dim, 128),
                    nn.ReLU()),
                nn.Sequential(
                    nn.Linear(128, latent_dim),
                    nn.ReLU()),
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    nn.ReLU())
            ]
            
            self.decoder_layers = [
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    nn.ReLU()),
                nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.ReLU()),
                nn.Sequential(
                    nn.Linear(128, hidden_dim),
                    nn.ReLU()),
                nn.Sequential(
                    nn.Linear(hidden_dim, self.output_dim))
            ]
            
            self.net = nn.Sequential(*(self.encoder_layers + self.decoder_layers))
        else:
            self.net = create_network(layer_dict, layer_constructors)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.net(x)
        return out.view(x.shape[0], self.window_size, self.input_dim)
    
    def summary(self):
        """ Return network summary """
        summary(self.net, (1, self.input_dim * self.window_size), device="cpu")
    
    def get_parameters(self):
        """ Get the number of parameters of the network """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class ConvAutoEncoder(nn.Module):
    def __init__(self, input_dim, create_network=None, layer_dict=None,  hidden_dim=256, latent_dim=64, output_dim=1, layer_constructors=None, window_size=50):
        super(ConvAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.input_shape = (input_dim, window_size)
        self.output_dim = input_dim * window_size
        self.window_size = window_size

        if layer_dict is None:
            self.encoder_layers = [
                nn.Sequential(
                    nn.Conv1d(input_dim, 32, kernel_size=3, stride=2, padding=1),
                    nn.ReLU()),
                nn.Sequential(
                    nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU()),
                nn.Sequential(
                    nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.ReLU()),
                nn.Sequential(
                    nn.Conv1d(128, latent_dim, kernel_size=3, stride=2, padding=1),
                    nn.ReLU())
            ]

            self.decoder_layers = [
                nn.Sequential(
                    nn.ConvTranspose1d(latent_dim, 128, kernel_size=4, stride=2, padding=1),
                    nn.ReLU()),
                nn.Sequential(
                    nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
                    nn.ReLU()),
                nn.Sequential(
                    nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU()),
                nn.Sequential(
                    nn.ConvTranspose1d(32, input_dim, kernel_size=4, stride=2, padding=1))
            ]
            self.net = nn.Sequential(*self.encoder_layers, *self.decoder_layers)

        else:
            self.net = create_network(layer_dict, layer_constructors)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, L, C] → [B, C, L]
        out = self.net(x)
        out = out.permute(0, 2, 1)  # [B, C, L] → [B, L, C]
        if out.size(1) > self.window_size:
            out = out[:, :self.window_size, :]

        return out


    def summary(self):
        """ Return network summary """
        summary(self.net, (self.window_size,self.input_dim), device="cpu")

    def get_parameters(self):
        """ Get the number of parameters of the network """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

def run_inference():
    """
    Script per eseguire l'inferenza di un modello PyTorch o ONNX all'interno di un container Docker.
    Accetta il percorso del modello, il percorso del file JSON di input e il percorso del file JSON di output.
    """
    if len(sys.argv) != 4:
        # Questo messaggio verrà visualizzato nei log del container se gli argomenti sono sbagliati
        print("Uso: python inference_script.py <model_path> <input_json_path> <output_json_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    input_json_path = sys.argv[2]
    output_json_path = sys.argv[3]

    print(f"DEBUG: Modello: {model_path}")
    print(f"DEBUG: Input JSON: {input_json_path}")
    print(f"DEBUG: Output JSON: {output_json_path}")    
    device = 'cpu'
    try:

        with open(input_json_path, 'r') as f:

            input_data = json.load(f)

        # check if the model is correctly initialized
        if model_path.lower().endswith(('.pt', '.pth')):
                    # PyTorch Model (TorchScript)
                    model = recreate_sequential_model_from_jit(model_path)
                    validation_result = validate_model_against_sdl_for_frontend(model)
                    saved_layers = NU.save_layers(model)
                    model = AutoEncoder(create_network=NU.reset_network, input_dim=6, window_size=50,layer_dict=saved_layers,layer_constructors=layer_constructors).to(device)
                    if validation_result['is_valid']:
                         print("Modello caricato correttamente",flush=True)
                    else:
                         raise ValueError(f"Il modello non è nel formato corretto: {validation_result['hints'], validation_result['invalid_blocks']}")
        else:
            raise ValueError(f"Formato modello non supportato: {os.path.splitext(model_path)[1]}")
        

        print(model_path.split('.')[0].split('/')[3])
        model_id = model_path.split('.')[0].split('/')[3]
        dataset_type = input_data['data']['dataset']
        metrics_required = input_data['data']['metrics']     
        train_loader, val_loader, test_loader= prepare_dataset_from_json(dataset_type)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        epochs = 100
        scaler = MinMaxScaler()

        device = 'cpu'

        print("Network Improver initialization ...",flush=True)

        start = time.time()

        net_improver = NI.ImproveNet(model, train_loader, val_loader, test_loader, criterion, optimizer, train_func,test_func, device, compute_metrics,scaler, test_loader,
                                metrics_required=metrics_required, epoches=20,max_divergence_attempts=10,max_overfitting_attempts=10,max_reduction_attempts=10,model_class=AutoEncoder,layer_constructors=layer_constructors)
        print(f"Network Improver initialized in {time.time() - start} s",flush=True)

        print('Starting Reduction...',flush=True)
        start = time.time()
        new_model_layer, reductionratio, inference, original_inference, original_size, modelsize = net_improver.check_performance()
        print(f"Reduction obtained in {time.time() - start} s",flush=True)
        
        OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimized_models")
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        model_path = os.path.join(OUTPUT_FOLDER, f"{model_id}_optimized.pt")
        torch.save(new_model_layer.state_dict(), model_path)
        print(f"[DEBUG] Modello salvato in: {model_path}", flush=True)


        # Salva i risultati in una struttura richiamabile da /execution-result
        results = {
            "optimized_model_path": f"/optimized_models/{model_id}_optimized.pt",
            "metrics": {
                "inference_time": inference,
                "original_inference_time": original_inference,
                "optimized_size": modelsize,
                "original_size": original_size,
                "reduction_ratio": reductionratio
            }
        }
        # model_results = {}

        # # 2. Carica ed esegui il modello
        # if model_path.lower().endswith(('.pt', '.pth')):
        #     # PyTorch Model (TorchScript)
        #     model = torch.jit.load(model_path)
        #     model.eval()
            
        #     # with torch.no_grad(): # Disabilita il calcolo dei gradienti per l'inferenza
        #     #     output = model(input_tensor)
            
        #     model_results = {"output": output.tolist()} # Converti il tensore di output in lista Python
            # print(f"DEBUG: Inferenza PyTorch completata. Output: {model_results['output'][:5]}...") # Mostra solo i primi 5 elementi
            
        # elif model_path.lower().endswith('.onnx'):
        #     # ONNX Model
        #     sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            
        #     # Ottieni i nomi degli input e output dal modello ONNX
        #     input_name = sess.get_inputs()[0].name
        #     output_name = sess.get_outputs()[0].name
            
        #     # ONNX Runtime lavora con array NumPy
        #     onnx_input = {input_name: input_tensor.numpy()} # Converti da tensore PyTorch a NumPy array
            
        #     output_data = sess.run([output_name], onnx_input)
        #     model_results = {"output": output_data[0].tolist()}
        #     print(f"DEBUG: Inferenza ONNX completata. Output: {model_results['output'][:5]}...") # Mostra solo i primi 5 elementi



        # 3. Salva i risultati dell'inferenza in un file JSON
        # Il formato di output qui è solo i risultati del modello
        # L'app.py aggiungerà 'status: completed' e 'results' come chiave superiore
        with open(output_json_path, 'w') as f:
            json.dump(results, f) 
        
        print(f"Inferenza completata. Risultati salvati in {output_json_path}")

    except Exception as e:
        # Gestione degli errori: scrivi l'errore nel file di output per informare il server
        # L'app.py leggerà questo per determinare lo stato 'execution_failed'
        error_info = {"error": "An error occured during inference", "status": "execution_failed"} # Usa 'execution_failed' come stato per coerenza
        with open(output_json_path, 'w') as f:
            json.dump(error_info, f)
        print(f"ERRORE durante l'inferenza: {e}")
        sys.exit(1) # Esce con un codice di errore

if __name__ == "__main__":
    print("PROVA",flush=True)
    run_inference()