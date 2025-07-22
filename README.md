# ImproveNet Service: Neural Network Compression as a Cloud–Edge Service

## About
**ImproveNet Service** is a lightweight and modular system that enables real-time neural network compression via a cloud–edge architecture. Edge devices can send their models to the server with a performance constraint (accuracy or loss), and receive a structurally reduced version ready for deployment — all through a simple asynchronous API.

Designed for constrained devices such as **Jetson Nano**, **Raspberry Pi 4**, and **Intel NUC**, the system supports CNNs and autoencoders and integrates seamlessly with real-world edge scenarios.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/improvenet-service.git
cd improvenet-service
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```


---

## Project Structure

```
.
├── backend/                   # Core backend logic and APIs
│   ├── app.py                 # Flask server - launch this to start the service
│   ├── client.py              # Script to simulate an edge device request
│   ├── controls.py            # Input validation and preprocessing
│   ├── inference_script.py    # Network compression logic interface
│   ├── prepare_dataset.py     # Dataset preprocessing based on selected domain
│
├── frontend/                  # Web-based frontend interface
│   ├── index.html             # Main interface
│   ├── script.js              # Logic for sending model and parameters
│   └── style.css              # UI styling
│
├── requirements.txt           # Python dependencies
└── README.md
```

---

##  How to Run the Server

Navigate to the backend directory and launch the Flask server:

```bash
cd backend
python app.py
```

The server will be ready to receive model compression requests via REST API.

---

## Client

You can simulate an edge device by running the included client:

```bash
cd backend
python client.py
```

This script sends a model along with desired performance constraints and retrieves the compressed version asynchronously.

---

## Web Interface

For a graphical interface, open `frontend/index.html` in a web browser. The frontend allows users to

- Upload a model
- Set the domain and metric
- Monitor the compression request and download the reduced model

---

## About ImproveNet

The core ImproveNet algorithm is **not included** in this repository due to privacy and IP restrictions. However, the backend is designed to integrate any compression engine by editing the `inference_script.py`, specifically the compression logic inside.

You may replace the placeholder logic with your own compression framework.

---

## Experimental Results

Experiments were conducted on AlexNet, VGG16, VGG19 and autoencoders using CIFAR-10 and ESA-ADB datasets. Results show up to 94.5% reduction in size and over 80% inference speed gain on edge devices like Raspberry Pi, Jetson Nano and Intel NUC. See the full paper for details.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

