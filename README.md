# V2X Dynamic Trust Management

## Overview

This project implements a **hybrid deep learning model** for **dynamic trust evaluation** in Vehicle-to-Everything (V2X) communication networks. Leveraging **Zero Trust Architecture (ZTA)** principles, it detects anomalies in Controller Area Network (CAN) bus data using a combination of **CNN** and **LSTM** models.

The project supports real-time cybersecurity in V2X communication by analyzing CAN data from the **ROAD dataset** to detect malicious behavior and improve **functional safety** in Intelligent Transportation Systems (ITS).

---

## Features

- CAN dataset preprocessing with resampling and interpolation
- Sliding window time-series data generation
- CNN for feature extraction, LSTM for sequential analysis
- Class imbalance handling with oversampling
- Evaluation metrics: Accuracy, Precision, Recall
- Lightweight, real-time detection model

---

## Dataset

### ROAD Dataset

- **Full Name**: Real ORNL Automotive Dynamometer (ROAD) Dataset
- **Purpose**: Benchmark dataset for CAN Intrusion Detection Systems (IDS)
- **Content**: Time-series CAN data under ambient and attack conditions
- **Labeling**:
  - `Label=0`: Normal (ambient) data
  - `Label=1`: Attack data
- **Format**: CSV files with signal values for each CAN ID over time

🔗 [ROAD Dataset Paper](https://arxiv.org/abs/2012.14600)

---

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment (optional)

### Dependencies

```bash
pip install numpy pandas scikit-learn tensorflow
