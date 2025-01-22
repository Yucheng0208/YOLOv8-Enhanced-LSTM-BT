# YOLOv8 Pose and Object LSTM Bluetooth Combined

## Introduction
This project integrates **YOLOv8** for pose detection and object recognition, **LSTM (Long Short-Term Memory)** for sequential analysis, and **Bluetooth communication** for real-time interaction. It is designed to provide a robust solution for tracking and analyzing pose and object data in dynamic environments.

## Quick Start

This is my HackMD.io post about the [YOLOv8 Tutorial](https://hackmd.io/@Yucheng208/YOLOv8-Tutorial). It serves as a quick guide to help you get started efficiently. The content is written in Chinese and is user-friendly for Chinese-speaking readers. Feel free to refer to it for your learning needs!

---

## Features
- **Pose Detection**: Utilizing YOLOv8 for real-time pose estimation.
- **Object Recognition**: Detects objects in frames with high accuracy.
- **Sequential Analysis**: Implements LSTM to analyze pose and object sequences for pattern recognition.
- **Bluetooth Integration**: Enables wireless data transfer for real-time applications.
- **Customizable and Scalable**: Easily adaptable for different use cases, including sports analytics, robotics, and more.

---

## TODO List: Bluetooth Integration
This section outlines the tasks required to integrate Bluetooth communication into the project.

1. **Bluetooth Module Configuration**
   - [ ] Install and configure the Bluetooth module on your device (e.g., HC-05, HC-06, or similar).
   - [ ] Could you verify the module is discoverable and properly paired with your computer or target device?

2. **Connection Protocol**
   - [ ] Define the communication protocol for transferring pose and object data (e.g., JSON or CSV format).
   - [ ] Implement handshaking to ensure reliable data transfer between the device and the application.

3. **Bluetooth API**
   - [ ] Utilize `PyBluez` or an equivalent library for Python-based Bluetooth communication.
   - [ ] Implement basic Bluetooth functions:
     - [ ] Device scanning
     - [ ] Pairing and unpairing
     - [ ] Sending and receiving data

4. **Integration with YOLOv8 and LSTM**
   - [ ] Update the `main.py` script to send YOLOv8 and LSTM outputs to the paired Bluetooth device.
   - [ ] Design a feedback mechanism to receive acknowledgment or additional commands via Bluetooth.

5. **Testing and Debugging**
   - [ ] Test Bluetooth communication in a controlled environment with static and dynamic data.
   - [ ] Debug any latency or data loss issues during transfer.

6. **Documentation**
   - [ ] Document all configurations, including device names, UUIDs, and pairing codes.
   - [ ] Provide a user guide for setting up the Bluetooth environment.

---

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Yucheng0208/YOLOv8_Pose_and_Object_LSTM_Bluetooth_Combined.git
   cd YOLOv8_Pose_and_Object_LSTM_Bluetooth_Combined
   ```

2. Create a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the environment:
   - Ensure that your device installs and configures the necessary Bluetooth modules.
   - Update the configuration file `config.yaml` with your device-specific parameters.

---

## Usage
1. Run the YOLOv8 pose and object detection module:
   ```bash
   python yolo_detection.py
   ```

2. Enable LSTM sequential analysis:
   ```bash
   python lstm_analysis.py
   ```

3. Start Bluetooth communication for data transfer:
   ```bash
   python bluetooth_transfer.py
   ```

4. Combine all modules:
   ```bash
   python main.py
   ```

---

## File Structure
```
YOLOv8_Pose_and_Object_LSTM_Bluetooth_Combined/
│
├── data/                   # Sample datasets and testing data
├── models/                 # YOLOv8 and LSTM pre-trained models
├── scripts/                # Custom utility scripts
├── yolo_detection.py       # YOLOv8 pose and object detection
├── lstm_analysis.py        # LSTM sequential analysis
├── bluetooth_transfer.py   # Bluetooth communication script
├── main.py                 # Main program combining all modules
├── config.yaml             # Configuration file for parameters
└── requirements.txt        # Required Python libraries
```


---

## License
This project is licensed under the [MIT License](LICENSE).
