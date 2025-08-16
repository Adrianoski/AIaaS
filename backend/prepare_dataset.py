import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

criterion = nn.MSELoss()
def create_windows(data, window_size, stride=1):
    """
    Args:
        data (numpy.ndarray): Dati di input.
        window_size (int): Lunghezza della finestra.
        stride (int): Passo di scorrimento tra le finestre.
    Returns:
        torch.Tensor: Tensore con i dati suddivisi in finestre.
    """
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i : i + window_size])
    return torch.tensor(np.array(windows), dtype=torch.float32)
class TrainingContext:
    def __init__(self):
        self.val_loader = None
        self.val_data = None
        self.train_data = None

ctx = TrainingContext()

def prepare_dataset_from_json(json_input, base_path='', window_size=50, stride=50, batch_size=128, device='cpu'):
    """
    Carica e prepara un dataset a partire da un JSON che contiene il nome del dataset.

    Args:
        json_input (dict): JSON deserializzato, deve contenere la chiave 'dataset'.
        base_path (str): Cartella base dove si trovano i dataset (es. 'datasets').
        window_size (int): Lunghezza delle finestre temporali.
        stride (int): Passo tra le finestre.
        batch_size (int): Dimensione del batch.
        device (str): 'cpu' o 'cuda'.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    dataset_name = json_input
    if not dataset_name:
        raise ValueError("Il campo 'dataset' è mancante nel JSON di input.")
    end_name = dataset_name + ".csv"
    file_path = os.path.join(base_path, "data", end_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset non trovato in: {file_path}")
    
    if dataset_name == 'esa_anomaly':
        data = pd.read_csv(file_path)
        channels = ['channel_41', 'channel_42', 'channel_43', 'channel_44', 'channel_45', 'channel_46']
        if not all(channel in data.columns for channel in channels):
            raise ValueError("Il dataset non contiene tutti i canali richiesti.")

        selected_data = data[channels].values
        scaler = MinMaxScaler()
        selected_data = scaler.fit_transform(selected_data)

        train_data, temp_data = train_test_split(selected_data, test_size=0.3, shuffle=False)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

        train_windows = create_windows(train_data, window_size, stride).to(device)
        val_windows = create_windows(val_data, window_size, stride).to(device)
        test_windows = create_windows(test_data, window_size, stride).to(device)

        train_loader = torch.utils.data.DataLoader(train_windows, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_windows, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_windows, batch_size=batch_size, shuffle=False)
        ctx.val_loader = val_loader
        ctx.val_data = val_data
        ctx.train_data = train_data
        return train_loader, val_loader, test_loader
    
    elif dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
        #transforms.Resize((227,227)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        #transforms.Resize((227,227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2023, 0.1994, 0.2010])
    ])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    ctx.val_loader = testloader
    ctx.val_data = testloader
    ctx.train_data = trainloader

def test_func(model,test_loader):
    outputs=[]
    with torch.no_grad():
            for test_batch in test_loader:
                output = model(test_batch)
                outputs.append(output)
            
    return outputs

def compute_metrics(prediction,ground_truth,criterion):
    "Implement your function here"
    loss = 0
    i=0
    for elem in ground_truth:
        loss += criterion(prediction[i], elem).item()
        i+=1
    loss = loss/len(ground_truth)
    diz = {'name':'loss', 'value': float(loss)}
    return [diz]

def train_func(model,train_loader,val_windows,optimizer, epochs=100):
    train_losses=[]
    val_losses=[]
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)  
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            batch_loss = 0
            for val_batch in ctx.val_loader:
                val_output = model(val_batch)
                val_loss = criterion(val_output, val_batch)
                batch_loss+=val_loss.item()
            val_losses.append(batch_loss/len(ctx.val_data))
        train_losses.append(epoch_loss/len(ctx.train_data))
        

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.6f}, Validation Loss: {val_loss.item():.6f}")

    return train_losses, val_losses