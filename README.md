# Machine Failure Detection — TinyML Project

## What Does This Project Do?

This project teaches a computer to **predict whether an industrial machine will fail**, using sensor data like temperature, vibration, and power usage. The trained model is tiny enough to run on a **microcontroller (ESP32)** — no internet or cloud needed. This is called **TinyML** (Tiny Machine Learning).

---

## How It All Fits Together

```
Your CSV Dataset
      │
      ▼
┌─────────────────┐
│  data_pipeline  │  ← Cleans, augments, and prepares data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    train.py     │  ← Trains the neural network
└────────┬────────┘
         │
         ├──── models/model.h5            (full Keras model)
         ├──── models/model.tflite        (lightweight version)
         ├──── models/model_quant.tflite  (INT8 version for ESP32)
         └──── models/preprocessor.pkl   (data transformer)
              
         ▼
┌─────────────────┐      ┌──────────────────────────────┐
│    infer.py     │  OR  │  edge_deployment/             │
│  (PC terminal)  │      │  esp32_inference.cpp (ESP32)  │
└─────────────────┘      └──────────────────────────────┘
```

---

## Project Files Explained

### 📄 `data_pipeline.py` — Data Preparation

This file takes your raw CSV and gets it ready for training. It does the following steps **automatically**:

| Step | What It Does | Why It Matters |
|------|-------------|----------------|
| **Load CSV** | Reads your dataset file | Starting point |
| **Detect target column** | Finds the "failure/no-failure" column automatically | Works with any CSV |
| **Drop ID columns** | Removes `id`, `product_id` etc. | IDs confuse the model |
| **Encode categories** | Converts text like `"Lathe"` → numbers | ML only understands numbers |
| **Scale numbers** | Normalizes values (e.g. temp 250°C → 0.83) | Prevents big numbers dominating |
| **Gaussian Noise** | Adds tiny random variation to minority class | Creates more training variety |
| **Feature Interpolation** | Blends pairs of minority samples | More natural-looking new data |
| **BorderlineSMOTE** | Generates synthetic failure samples near the decision edge | Improves failure detection |
| **Train/Test Split** | 80% training, 20% testing | Measures real-world accuracy |

> **Why augmentation?** Machine failure datasets are almost always **imbalanced** — 95% normal, 5% failure. Without augmentation, the model just learns to say "normal" every time and gets 95% accuracy while being useless. The augmentation pipeline makes the training data larger and more balanced.

---

### 📄 `train.py` — Training the Model

This file builds and trains the neural network.

#### The Neural Network Architecture

```
Input (sensor features)
        │
  Dense(32 neurons)     ← learns complex patterns
  BatchNormalization     ← stabilizes training
  ReLU activation       ← adds non-linearity
  Dropout(30%)          ← randomly disables neurons to prevent memorizing
        │
  Dense(16 neurons)     ← refines patterns
  BatchNormalization
  ReLU
  Dropout(20%)
        │
  Dense(1, Sigmoid)     ← outputs 0.0 (normal) to 1.0 (failure)
```

Think of neurons like detectives — each one looks for a specific pattern in the sensor data. More neurons = more patterns detected = better predictions.

#### Training Techniques Used

| Technique | Simple Explanation |
|-----------|-------------------|
| **Class Weights** | Tell the model "failed machines are rare but more important" |
| **Early Stopping** | Stops training when the model stops improving (patience = 20 epochs) |
| **Cosine LR Decay** | Gradually slows down learning to fine-tune the model |
| **ReduceLROnPlateau** | Halves the learning rate when stuck |
| **ModelCheckpoint** | Saves the best version of the model automatically |

#### Output Files After Training

| File | Description |
|------|-------------|
| `models/model.h5` | Full Keras model — use on PC |
| `models/model.tflite` | Lightweight version — smaller, faster |
| `models/model_quant.tflite` | INT8 quantized — for ESP32/microcontrollers |
| `models/best_model.keras` | Best checkpoint saved during training |
| `models/preprocessor.pkl` | Remembers how data was scaled/encoded |
| `models/categorical_values.pkl` | Valid options for each category column |

---

### 📄 `infer.py` — Predicting on PC

After training, run this to test the model interactively. It will ask you to type in sensor values, then print a prediction.

```
Please enter numerical sensor values:
  Temperature: 95
  Vibration: 62
  Power_Usage: 11

Please enter categorical values:
  Machine_Type (options: Drill / Lathe / Mill): Lathe

===============================
 Prediction: FAILURE 🚨
 Confidence: 87.3%
===============================
```

The valid options shown (e.g., `Drill / Lathe / Mill`) come directly from your training dataset — no hardcoded values.

---

### 📄 `edge_deployment/esp32_inference.cpp` — Running on ESP32

This C++ file lets the trained model run **directly on an ESP32 microcontroller** — no PC, no internet.

**How it works on ESP32:**
1. The `model_quant.tflite` file is converted to a C header file (`model_data.h`) using:
   ```bash
   xxd -i model_quant.tflite > model_data.h
   ```
2. The C++ code loads this model into the ESP32's memory (4KB arena)
3. Sensor readings are fed into the model every 2 seconds
4. The result is printed to the Serial monitor: `NORMAL ✅` or `FAILURE DETECTED 🚨`

**Why INT8 quantization?** The ESP32 has very limited memory. INT8 uses 4× less memory than float32, making the model fit on a microcontroller without losing much accuracy.

---

## How to Run the Project

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the Model
```bash
cd machine_failure_tinyml
python train.py --csv "path\to\your_dataset.csv"
```

### Step 3 — Run Interactive Inference on PC
```bash
python infer.py
```

### Step 4 — Run the Digital Twin Simulation
See the model detect impending machine failure in real-time! This console simulator replays true historical "normal" sensor data from your dataset every 2 seconds, runs them through the neural network, and automatically injects a massive anomaly spike once every minute to demonstrate live failure detection.

**Optional: AI Incident Reports**
You can provide a Gemini API key. When a failure is detected, the simulator will automatically pause and generate an AI root-cause analysis and resolution report right in your terminal!
```bash
python digital_twin.py --csv "path\to\your_dataset.csv" --api-key "YOUR_GEMINI_API_KEY"
```

### Step 5 — Deploy to ESP32 (optional)
```bash
# Convert model to C array
xxd -i models/model_quant.tflite > edge_deployment/model_data.h

# Open edge_deployment/esp32_inference.cpp in Arduino IDE
# Install TensorFlowLite_ESP32 library
# Flash to ESP32
```

---

## What the Dataset Should Look Like

The project works with any CSV that has:
- **Numerical columns**: Temperature, Vibration, Power Usage, etc.
- **Categorical columns**: Machine type, location, etc.  
- **A target column** named something like `failure`, `target`, `status`, or `is_failure` with values `0` (normal) and `1` (failure)

Example:

| Temperature | Vibration | Power_Usage | Machine_Type | failure |
|-------------|-----------|-------------|--------------|---------|
| 72.5 | 35.2 | 8.1 | Lathe | 0 |
| 95.1 | 78.4 | 14.3 | Drill | 1 |

---

## How Accuracy Was Improved

| What Changed | Before | After |
|---|---|---|
| Augmentation | SMOTE only | Gaussian Noise + Mixup + BorderlineSMOTE |
| Hidden layers | 1 layer (8 neurons) | 2 layers (32 → 16 neurons) |
| Regularization | None | BatchNorm + Dropout |
| Imbalance handling | SMOTE | SMOTE + Class Weights |
| LR schedule | Step-wise | Cosine decay + Plateau backup |
| Model saving | End only | Best checkpoint saved automatically |

---

## Key Concepts Glossary

| Term | Simple Explanation |
|------|-------------------|
| **TinyML** | Machine learning that runs on tiny devices (no cloud needed) |
| **Neural Network** | A system inspired by the brain; learns patterns from examples |
| **Epoch** | One full pass through all training data |
| **Overfitting** | Model memorizes training data, fails on new data; fixed by Dropout |
| **SMOTE** | Creates fake minority samples by interpolating between real ones |
| **BatchNorm** | Normalizes neuron outputs to make training faster and stable |
| **INT8 Quantization** | Compresses model weights from 32-bit to 8-bit integers; 4× smaller |
| **TFLite** | TensorFlow Lite — a compressed format for running ML on edge devices |
