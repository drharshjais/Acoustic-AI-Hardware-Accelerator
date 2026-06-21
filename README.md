# Acoustic AI Hardware Accelerator for Deforestation Detection 🌲💻

## Overview
This project is a complete Hardware-Software Co-Design system built to detect illegal logging operations in real-time. It features a custom 8-bit quantized Convolutional Neural Network (CNN) deployed onto a custom-designed Verilog ASIC architecture. 

Unlike standard software-only AI models, this project proves the end-to-end viability of baking machine learning directly into low-power silicon for remote smart city deployments.

## System Architecture

### 1. The Hardware (Digital Logic / VLSI)
Designed in Verilog and simulated in Xilinx Vivado, the custom ASIC features:
* **FSM Control Unit:** Synchronizes memory fetching and computation.
* **Multiply-Accumulate (MAC) Engine:** A custom 32-bit arithmetic core simulating a neural network processing element.
* **BRAM & ROM Integration:** Manages the 10,048-pixel acoustic footprint and static `.mem` weights.
* **Hardware Trigger Logic:** A comparator circuit that physically pulls a digital pin HIGH upon threat detection, designed to interface with a LoRaWAN radio.

### 2. The Software (Machine Learning / DSP)
* **Digital Signal Processing:** Python pipeline utilizing `librosa` to convert raw audio into mathematically bound Mel Spectrograms.
* **Quantized AI:** A CNN trained in TensorFlow, mathematically quantized to INT8 format for extreme power efficiency on edge devices.
* **Software-in-the-Loop Bypass:** A diagnostic pipeline engineered to bypass OS-level audio distortion, injecting pure dataset vectors directly into the inference engine to guarantee mathematical alignment with the hardware MAC unit.

## Repository Structure
* `/hardware` - Verilog source files (`.v`) for the RAM, ROM, FSM, and MAC unit.
* `/software` - Python DSP scripts and the `.tflite` quantized model.
* `/docs` - System architecture diagrams and Vivado simulation waveforms (`.vcd`).

## Developer
**Harsh Jaiswal** *B.Tech, Electronics and Communication Engineering*
