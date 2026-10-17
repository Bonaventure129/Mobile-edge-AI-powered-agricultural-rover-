# ♻️ SmartBin: Autonomous Edge-AI Waste Classifier

**An intelligent, computer-vision-powered sorting system for automated waste management.**

## 📖 Executive Summary
The **SmartBin** is an automated, edge-computing hardware system designed to classify and sort waste in real-time. Utilizing a lightweight computer vision model deployed on a Raspberry Pi, the system identifies the type of waste presented to its camera and uses WebSocket protocols to trigger a microcontroller (e.g., ESP32/Arduino) to open the correct disposal compartment (e.g., Recyclable vs. Non-Recyclable).

By running a quantized TensorFlow Lite model directly at the edge, the SmartBin operates with ultra-low latency and requires no active cloud computing or continuous internet connection to perform its core sorting functions.

## ✨ Core Capabilities
* **Real-Time Edge Inference:** Evaluates video frames locally using a TensorFlow Lite (`tflite-runtime`) model optimized for ARM64 CPUs.
* **Computer Vision Pipeline:** Leverages OpenCV (`opencv-python`) to capture, crop, and pre-process hardware camera feeds efficiently.
* **High-Speed Actuation:** Uses persistent WebSocket connections (`websocket-client`) to instantly transmit classification results to a kinematic node (microcontroller) for immediate servo/motor actuation.
* **Isolated Virtual Environment:** Clean, conflict-free Python deployment using PEP 668 compliant `venv` architecture.

## ⚙️ System Architecture

### Software Stack
* **Operating System:** Debian Linux (Raspberry Pi OS)
* **Vision / Capture:** OpenCV (`cv2`)
* **Machine Learning Engine:** TensorFlow Lite Runtime (`tflite-runtime`)
* **Matrix Mathematics:** NumPy
* **Communication Protocol:** WebSockets

### Hardware Workflow (Expected)
1. **Input:** Object is presented to the Raspberry Pi Camera.
2. **Processing:** The Pi captures the frame, pre-processes the matrix, and runs the TFLite inference.
3. **Transmission:** The dominant classification class is sent over a local WebSocket payload.
4. **Actuation:** A receiving microcontroller parses the payload and actuates the corresponding physical bin lid.

## 🚀 Installation & Setup

This project requires Python 3 and a fully isolated virtual environment to prevent package conflicts on modern Debian/Raspberry Pi systems.

### 1. Create and Activate the Virtual Environment
Navigate to your project directory and establish the environment:
```bash
cd ~/smartbin_dir
python3 -m venv binVenv
source binVenv/bin/activate
```

### 2. Install Dependencies
With the virtual environment active (indicated by `(binVenv)` in your terminal prompt), install the required packages. 

*Note: A timeout extension is recommended for slower Wi-Fi connections.*
```bash
pip install --default-timeout=100 opencv-python numpy tflite-runtime websocket-client
```

## 🛠️ Usage Instructions

### Running the Classifier
Ensure your virtual environment is active, then execute the main Python script:
```bash
# Activate environment if opening a fresh terminal
source ~/smartbin_dir/binVenv/bin/activate

# Launch the vision and communication loop
python main.py
```

## 📝 Configuration & Customization
To adapt this system to your specific machine learning model, ensure the following parameters in your main script match your training data:
* **Model Path:** Point the `tflite.Interpreter` to your exported `.tflite` model file.
* **Label Map:** Update your `CLASSES` array to reflect your specific waste categories (e.g., `["Plastic", "Paper", "Metal", "Organic", "Unknown"]`).
* **WebSocket IP:** Ensure the target URI matches the IP address of your ESP32/Microcontroller (e.g., `ws://192.168.4.1:81`).

***
*Engineered and deployed for automated, resource-efficient hardware-in-the-loop IoT infrastructure.*