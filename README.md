# HALSP-Net: A Shared Projection Architecture with Dynamic Channel Selection 
### Hierarchical Active channel Latent Shared Projection Network

This repository contains the introduction and implementation details for the paper **"HALSP-Net: A Shared Projection Architecture with Dynamic Channel Selection"**.

## Abstract

Efficient inference and on device training under tight parameter, compute, and memory budgets are recurring needs across multimedia signal processing pipelines. We propose a convolutional architecture in which a single learnable weight matrix per stage carries three roles, namely entry projection, inner channel mixing, and exit projection, and is paired with a per block depthwise spatial filter. Channel mixing and spatial processing are kept as separate operations, and during training only a dynamic subset of latent channels is active at each step while inference uses the dense weight matrix. This combination of weight sharing across roles, decoupled spatial and channel paths, and selective channel activation keeps parameter and memory cost low. It does so without giving up the headroom that larger designs use for accuracy. On CIFAR-100, used as the evaluation testbed, the proposed model reaches accuracy comparable to deep residual and wide residual networks while staying well below two million parameters. On a single high end GPU it sustains the highest single image throughput in our panel. The dynamic active channel slice lets the same model fit training batches at which several comparable accuracy baselines run out of memory. This exposes a controllable training time speed and memory trade off with negligible loss in final accuracy.

## Web

* [Paper Details](#)

## Introduction

Compact convolutional designs are usually built by reducing parameters, multiply add operations, or activation memory needed at training and inference time. Our design keeps sparsity entirely on the training side and runs inference on the dense weight matrix. This direct targeting of resource cost is a recurring theme in multimedia signal processing, where adaptive inference and lightweight feature extraction have been studied.

**What the method does:**
Instead of using separate weight matrices for consecutive linear transformations in every bottleneck block, the method uses a **single weight matrix ($W$)** per stage in three roles, namely entry, inner, and exit. It pairs this matrix with a per block depthwise spatial filter, so that spatial and channel paths are kept separate at every block.

* Original shared matrix idea is inspired by: [halvit](https://github.com/sp4cing-itu/halvit)

## Method

The proposed HALSP architecture directly targets efficiency by combining three principles:

1. **Weight Sharing Across Roles:** A single learnable weight matrix $W$ per stage spans the full output space. It acts in three distinct roles: entry projection, inner channel mixing (using cyclic column shifts per block), and exit projection.
2. **Decoupled Spatial Processing:** Channel mixing and spatial processing are explicitly separated. The shared matrix $W$ handles only the channel mixing, while a dedicated per-block depthwise spatial filter manages the spatial processing.
3. **Dynamic Channel Selection Mechanism:** During training, only a dynamic subset of latent channels is active at each step, controlled by the active channel fraction ($r_a$). This mechanism operates by rebuilding the active subset every $T$ steps:
    * **Topology Refresh:** Channels are scored by their mean absolute weights (or optimizer momentum) and split into a **Focus Pool** (top-scoring channels) and a **Reserve Pool** (the remaining channels).
    * **Exploit Branch:** Uniformly samples a portion of the active channels directly from the Focus Pool.
    * **Opportunity Map ($Q$):** The network maintains an Exponential Moving Average (EMA) of the input variance trace ($v$). It calculates the "Active Coverage" ($K$), which measures how much of the input variance is already covered by the current active channels in the Focus Pool. The Opportunity Map is then defined as $Q = v / (K + \epsilon)$. Essentially, $Q$ identifies the "uncovered" variance—the missing features or learning opportunities that the active channels are failing to capture.
    * **Explore Branch:** Channels from the Reserve Pool are scored based on their cosine similarity with the Opportunity Map $Q$. A multinomial sample is drawn to select reserve channels that best match these missing features, ensuring the network explores effectively.
    * *Inference:* This sparse selection is strictly a train-side operation. Inference relies completely on the full, dense weight matrix, ensuring no accuracy is lost from the dynamic slicing.

## Results

The following tables present the exhaustive benchmark results of the HALSP configurations evaluated on the CIFAR-100 dataset using a single NVIDIA A100 40GB GPU.

### Classification Accuracy (CIFAR-100)

| Model Configuration | Params (M) | Augmentation | Active Fraction ($r_a$) | Top-1 Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- |
| HALSP (3,4,6,3) | 1.71 | Basic | 0.5 | 74.82 |
| HALSP (3,4,6,3) | 1.71 | TA+LS | 0.1 | 77.19 |
| HALSP (3,4,6,3) | 1.71 | TA+LS | 0.5 | 78.65 |
| HALSP (3,4,6,3) | 1.71 | TA+LS | 1.0 (Dense) | 79.03 |
| HALSP (3,4,23,3) | 1.75 | Basic | 0.5 | 75.70 |
| HALSP (3,4,23,3) | 1.75 | TA+LS | 0.1 | 77.06 |
| HALSP (3,4,23,3) | 1.75 | TA+LS | 0.5 | 79.39 |
| HALSP (3,4,23,3) | 1.75 | TA+LS | 1.0 (Dense) | 79.41 |

*(Note: TA = TrivialAugment, LS = Label Smoothing)*

### Inference Performance (224x224 Resolution)

| Model | Batch Size | Throughput (images/s) | Latency (ms) | Peak VRAM (MB) |
| :--- | :--- | :--- | :--- | :--- |
| **HALSP 50** | 1 | 308.0 | 3.25 | 33.8 |
| **HALSP 50** | 16 | 3849.2 | 4.16 | 303.5 |
| **HALSP 50** | 128 | 4372.5 | 29.30 | 2296.6 |
| **HALSP 50** | 256 | 4424.2 | 57.90 | 4574.6 |
| **HALSP 101** | 1 | 220.1 | 4.54 | 34.0 |
| **HALSP 101** | 16 | 3296.3 | 4.85 | 303.7 |
| **HALSP 101** | 128 | 3814.3 | 33.60 | 2296.8 |
| **HALSP 101** | 256 | 3855.2 | 66.40 | 4574.8 |

### Single Image Inference FPS (Resolution Scaling at Batch=1)

| Model | 224x224 FPS | 512x512 FPS | 1088x1088 FPS |
| :--- | :--- | :--- | :--- |
| **HALSP 50** | 306.48 | 304.30 | 165.62 |
| **HALSP 101** | 218.67 | 217.36 | 146.38 |

### Training Step Performance

| Model | Active Rat. ($r_a$) | Batch Size | Throughput (images/s) | Latency (ms) | Peak VRAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HALSP 50** | 0.1 | 8 | 631.8 | 12.7 | 554.2 |
| **HALSP 50** | 0.1 | 16 | 1130.1 | 14.2 | 1084.1 |
| **HALSP 50** | 0.1 | 128 | 1282.3 | 99.8 | 8435.0 |
| **HALSP 50** | 0.1 | 512 | 1301.4 | 393.4 | 33631.4 |
| **HALSP 50** | 0.5 | 8 | 618.3 | 12.9 | 576.9 |
| **HALSP 50** | 0.5 | 16 | 1044.9 | 15.3 | 1121.5 |
| **HALSP 50** | 0.5 | 128 | 1215.0 | 105.4 | 8637.5 |
| **HALSP 50** | 0.5 | 512 | 1234.4 | 414.8 | 34428.8 |
| **HALSP 50** | 1.0 (Dense) | 8 | 589.2 | 13.6 | 606.3 |
| **HALSP 50** | 1.0 (Dense) | 16 | 1027.7 | 15.6 | 1168.6 |
| **HALSP 50** | 1.0 (Dense) | 128 | 1177.5 | 108.7 | 8889.5 |
| **HALSP 50** | 1.0 (Dense) | 512 | 1196.6 | 427.9 | 35420.7 |
| **HALSP 101** | 0.1 | 8 | 451.3 | 17.7 | 561.8 |
| **HALSP 101** | 0.1 | 16 | 915.2 | 17.5 | 1098.9 |
| **HALSP 101** | 0.1 | 128 | 1254.4 | 102.1 | 8552.4 |
| **HALSP 101** | 0.1 | 512 | 1279.6 | 400.1 | 34100.9 |
| **HALSP 101** | 0.5 | 8 | 436.4 | 18.3 | 617.0 |
| **HALSP 101** | 0.5 | 16 | 866.1 | 18.5 | 1201.2 |
| **HALSP 101** | 0.5 | 128 | 1115.2 | 114.8 | 9259.6 |
| **HALSP 101** | 0.5 | 512 | 1136.6 | 450.5 | 36910.9 |
| **HALSP 101** | 1.0 (Dense) | 8 | 425.8 | 18.8 | 690.1 |
| **HALSP 101** | 1.0 (Dense) | 16 | 851.7 | 18.8 | 1328.8 |
| **HALSP 101** | 1.0 (Dense) | 128 | 1027.4 | 124.6 | 10143.9 |
| **HALSP 101** | 1.0 (Dense) | 512 | OOM | - | - |

## Repository Contents

* `halsp.py`: This module implements the core `HalspNetStage` and a `ResNet50` wrapper. It contains the logic for the shared projection matrix, decoupled spatial processing, and the dynamic channel selection mechanism.
* `halsp_train.py`: The main training script for HALSP-Net on CIFAR-100. It implements the training recipe including learning rate scheduling, data augmentation (TrivialAugment), and phase management.
* `performance_test_inference.py`: A script to benchmark inference throughput and latency across different batch sizes for HALSP and various baseline models. It includes hardware telemetry monitoring.
* `performance_test_fps_resolutions.py`: A script to evaluate single-image FPS and FLOPs across different input resolutions for the panel of models.
* `performance_test_train.py`: A script to measure the training throughput, latency, and peak VRAM across different batch sizes, useful for observing the impact of the active channel slice ($r_a$) on memory and speed.
* `logs/`
  * `performance_logs/`
    * `performance_result_inference.txt`: Contains generated text reports detailing the inference benchmark results.
    * `fps_result_inference.txt`: Contains generated text reports detailing the resolution FPS benchmark results.
    * `performance_result_train.txt`: Contains generated text reports detailing the training benchmark results.
  * `train_logs/`
    * `r50_block_settings_0_50sparsity.txt`: Contains the detailed step-by-step training logs, accuracy metrics, and class-specific performance outputs for the HALSP-50 model with $r_a=0.5$.
    * `r101_block_settings_0_50sparsity.txt`: Contains the detailed step-by-step training logs, accuracy metrics, and class-specific performance outputs for the HALSP-101 model with $r_a=0.5$.

## Reference

A. N. Yılmaz and B. U. Töreyin, "HALSP-Net: A Shared Projection Architecture with Dynamic Channel Selection."

Signal Processing for Computational Intelligence Research Group (SP4CING).
