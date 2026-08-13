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
- PaddlePaddle >= 3.0.0 (recommend manual installation for GPU support)
- Ocean framework (PaddlePaddle Lightning)

### Setup

```bash
# Clone repository
git clone https://github.com/PlumBlossomMaid/Conan.git
cd Conan

# Install dependencies (excluding PaddlePaddle)
pip install -r requirements.txt

# Install PaddlePaddle manually
# For CUDA 11.8:
python -m pip install paddlepaddle-gpu==3.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

## Quick Start

### 1. Data Preparation

```bash
# Prepare your dataset with the following structure:
# data/
#   train/
#     speaker1/
#       001.wav
#       002.wav
#   valid/
#     speaker2/
#       001.wav

# Extract HuBERT embeddings and binarize
python konan/scripts/binarize_conan.py \
  --data_dir data/train \
  --output_dir data/train_binary \
  --num_workers 8
```

### 2. Training

Conan uses a unified training entry point with configuration files:

```bash
# Stage 1: Content Extractor (80k steps)
python konan/scripts/train.py -c configs/content_extractor.yaml

# Stage 2: Main Model with frozen content extractor (160k steps)
python konan/scripts/train.py -c configs/main.yaml

# Stage 3: Causal Shuffle Vocoder (600k steps)
python konan/scripts/train.py -c configs/vocoder.yaml
```

### 3. Inference

```bash
# Streaming inference
python konan/scripts/infer_conan.py \
  --source_audio source.wav \
  --reference_audio reference.wav \
  --output_audio output.wav \
  --content_ckpt checkpoints/content_extractor/best.pdparams \
  --main_ckpt checkpoints/conan_main/best.pdparams \
  --vocoder_ckpt checkpoints/vocoder/best.pdparams \
  --chunk_size 80  # ms
```

## Configuration

All training configurations use YAML files in `configs/`. Key parameters:

- `task_cls`: Model class to instantiate
- `work_dir`: Checkpoint save directory
- `pretrained`: Pretrained checkpoints and frozen parameters
- `max_batch_frames`: Dynamic batching for consistent GPU memory
- `use_distributed_sampler`: Enable distributed training

Example configuration structure:

```yaml
task_cls: konan.models.conan_main.ConanMainModel
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
├── konan/
│   ├── layers/               # Core model components
│   │   ├── stream_content_extractor.py
│   │   ├── adaptive_style_encoder.py
│   │   └── causal_shuffle_vocoder.py
│   ├── models/               # Training models (Ocean Lightning)
│   │   ├── content_extractor.py
│   │   ├── conan_main.py
│   │   └── vocoder.py
│   ├── scripts/              # CLI tools
│   │   ├── train.py          # Unified training entry
│   │   ├── binarize_conan.py
│   │   └── infer_conan.py
│   └── utils/                # Utilities
│       ├── indexed_datasets.py
│       ├── training_utils.py
│       └── model_utils.py
├── configs/                  # Training configurations
│   ├── content_extractor.yaml
│   ├── main.yaml
│   └── vocoder.yaml
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
