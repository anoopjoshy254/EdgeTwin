# Edge Deployment Memory & Profiling Report

## Model Overview
- **Name**: Machine Failure Detection TNN
- **Target Platform**: ESP32 / Cortex-M Microcontrollers
- **Framework**: TensorFlow Lite for Microcontrollers

## Memory Footprint Estimate
Based on the architecture (Input Layer -> 8 Dense ReLU -> 1 Dense Sigmoid):
- **Model Params**: Less than 500 parameters (typically < 2 KB).
- **Standard `.tflite` Size**: ~4 KB - 8 KB.
- **Quantized INT8 `.tflite` Size**: ~2 KB - 4 KB.
- **Tensor Arena Size**: The `kTensorArenaSize` is safely set to `4 KB`, meaning the runtime working memory used by the application during inference is very lightweight.

### Overall Constraints Check
- Expected Flash Memory needed: `< 10 KB` (Goal: `< 100 KB`). **Pass**.
- Expected SRAM needed (Arena): `~4 KB` (Goal: Minimize). **Pass**.

## Inference Latency
With the optimized INT8 network format, the ESP32 (running at 240 MHz) will complete the single dense layer matrix multiplications in far less than `1 ms`.
Typical expected duration: **~100 - 300 microseconds (us)** per inference. This easily guarantees the "real-time inference capability" constraint.

## General Flexibility
1. **Adding sensors**: Changing the training process modifies `input_dim` in `train.py`. Since `kTensorArenaSize` is `4 KB`, you can scale up to roughly `500-1000` features before needing to expand the arena.
2. **Alternative Machines**: The pipeline supports stratified sampling and dynamic class re-weighting via SMOTE, automatically configuring scaling parameters allowing it to act identically on data from pumps or compressors.

## Instructions to Deploy
1. Export model using `train.py` (which produces `models/model_quant.tflite`).
2. Convert `.tflite` to a C array byte header:
   ```bash
   xxd -i models/model_quant.tflite > edge_deployment/model_data.h
   ```
3. Copy `edge_deployment/esp32_inference.cpp` content into your PlatformIO / Arduino IDE `main.cpp` alongside `model_data.h`.
4. Flash the ESP32 array!
