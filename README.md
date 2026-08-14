# Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion

A chunkwise online network for zero-shot adaptive voice conversion with real-time streaming capability.

## Overview

Conan is a three-stage streaming voice conversion system that preserves linguistic content while transferring speaker characteristics in real-time. The system achieves **37ms latency on A100 GPU** with high-quality output.

### Key Features

- 🎯 **Zero-shot voice conversion**: No speaker-specific fine-tuning required
- ⚡ **Real-time streaming**: Chunkwise processing with ~37ms latency
- 🔄 **Fully causal architecture**: All components support online inference
- 🎨 **Adaptive style encoding**: Captures emotion and prosody via clustering VQ
- 🔊 **High-quality synthesis**: Causal Shuffle Vocoder with pixel shuffle upsampling

### Architecture

Conan consists of three core components:

1. **Stream Content Extractor (SCE)**: Emformer-based streaming encoder distilled from HuBERT
   - Chunk size: 4 frames (80ms @ 50Hz)
   - Left context: 1, Right context: 2
   - Output: 256-dim continuous content embeddings

2. **Adaptive Style Encoder (ASE)**: Clustering VQ with align attention
   - CVQ codebook: 128 codes × 64 dims
   - Captures fine-grained style (emotion, prosody)
   - Aligns style to content timeline

3. **Causal Shuffle Vocoder (CSV)**: Fully causal HiFiGAN
   - Replaces transposed convolutions with pixel shuffle
   - Eliminates checkerboard artifacts
   - Maintains strict causality for streaming

## Installation

### Prerequisites

- Python 3.8+
- PaddlePaddle >= 3.3 (install manually for your accelerator/runtime)
- Ocean framework

### Setup

```bash
# Clone repository
git clone https://github.com/PlumBlossomMaid/Conan.git
cd Conan

# Install project dependencies. PaddlePaddle is intentionally not included.
pip install -r requirements.txt

# Install PaddlePaddle manually for your environment:
# https://www.paddlepaddle.org.cn/install/quick
```

## Quick Start

### 1. Data Preparation

```bash
# Extract and flatten LibriTTS audio
python entry/preprocess.py -c configs/preprocess_libritts.yaml

# Extract HuBERT distillation targets with one ONNX session and batched inputs
python entry/preprocess.py -c configs/preprocess_hubert.yaml
```

### 2. Training

Conan uses a unified training entry point with configuration files:

```bash
# Stage 1: Distill HuBERT into the Stream Content Extractor
python entry/train.py -c configs/content_extractor.yaml

# Stage 2: Train the Causal Shuffle Vocoder
python entry/train.py -c configs/vocoder.yaml

# Stage 3: Train the main conversion model with the frozen content extractor
python entry/train.py -c configs/main.yaml
```

### 3. Inference

```bash
# Streaming inference
python entry/infer_conan.py \
  --source source.wav \
  --reference reference.wav \
  --output output.wav \
  --content-ckpt checkpoints/content_extractor/best.pdparams \
  --main-ckpt checkpoints/conan_main/best.pdparams \
  --vocoder-ckpt checkpoints/vocoder/best.pdparams \
  --streaming \
  --chunk-ms 80
```

## Configuration

All preprocessing and training configuration lives in YAML files under `configs/`. Key parameters:

- `data.wavs_dir`: Flattened waveform directory
- `data.hubert_onnx`: Batched HuBERT ONNX model path
- `preprocessing.max_batch_size`: ONNX batch size for HuBERT extraction
- `data.n_valid` / `data.valid_seed`: Random validation sample count and split seed
- `task_cls`: Model class to instantiate
- `max_batch_frames`: Dynamic batching for consistent GPU memory

Example configuration structure:

```yaml
task_cls: models.conan_main.ConanMainModel
work_dir: checkpoints/conan_main

pretrained:
  content_extractor: checkpoints/content_extractor/best.pdparams
  frozen_params:
    - content_extractor  # Freeze during fine-tuning
```

## Model Details

### Content Extraction

Uses **MSE regression** to distill HuBERT's continuous embeddings (256-dim) rather than K-Means clustering. This approach:
- Preserves fine-grained content information
- Avoids codebook collapse issues
- Simplifies training pipeline

### Streaming Architecture

All components maintain **causal dependencies**:
- Stream Content Extractor: Emformer with right-context caching
- Adaptive Style Encoder: Causal attention with memory bank
- Causal Shuffle Vocoder: Pixel shuffle for temporal upsampling

### Distributed Training

Built-in support for multi-GPU training:
- `DsBatchSampler`: Dynamic batching by total frames
- Sort by similar length for efficient padding
- Automatic gradient accumulation

## Project Structure

```
Conan/
├── layers/                   # Core model components
│   ├── stream_content_extractor.py
│   ├── adaptive_style_encoder.py
│   └── causal_shuffle_vocoder.py
├── models/                   # Training models (Ocean)
│   ├── content_extractor.py
│   ├── conan_main.py
│   └── vocoder.py
├── entry/                    # CLI entry points
│   ├── preprocess.py         # Unified preprocessing entry
│   ├── train.py              # Unified training entry
│   └── infer_conan.py
├── utils/                    # Utilities
│   ├── indexed_datasets.py
│   ├── training_utils.py
│   └── model_utils.py
├── configs/                  # Preprocessing and training configurations
│   ├── preprocess_libritts.yaml
│   ├── preprocess_hubert.yaml
│   ├── content_extractor.yaml
│   ├── vocoder.yaml
│   └── main.yaml
└── requirements.txt
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhang2025conan,
  title={Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion},
  author={Zhang, Yu and Tian, Baotong and Duan, Zhiyao},
  journal={arXiv preprint arXiv:2507.14534},
  year={2025}
}
```

## Paper

📄 [arXiv:2507.14534](https://arxiv.org/abs/2507.14534)

🎧 [Demo samples](https://arxiv.org/abs/2507.14534)

## License

[MIT License](LICENSE)

## Acknowledgments

- Software engineering inspired by [DiffSinger](https://github.com/openvpi/DiffSinger)
- HuBERT feature extraction from [fairseq](https://github.com/facebookresearch/fairseq)
- HiFi-GAN architecture from [official implementation](https://github.com/jik876/hifi-gan)
