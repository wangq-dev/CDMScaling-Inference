# VLM Inference Code

## Introduction

This repository provides code adaptations and a single-image GPU inference script for vision-language models (VLMs) trained using the [LLaVA-OneVision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/) framework. It supports different language model backbones by providing modified implementations for the official [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) codebase.

Our trained VLMs use [SigLIP-SO400M-Patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) as the vision encoder and are trained on selected portions of [Infinity-MM](https://huggingface.co/datasets/BAAI/Infinity-MM). See the corresponding model card for backbone-specific model details and training data information.

## How to Use

### Step 1. Set Up the Environment

Create a conda environment and install the required dependencies:

```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install -r requirements.txt
```

### Step 2. Install LLaVA-NeXT

Clone the official LLaVA-NeXT repository and install the package:

```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install -e .
cd ..
```

### Step 3. Apply Model Adaptations

Replace the corresponding files in the cloned LLaVA-NeXT repository with the modified implementations provided in this project.

Run the following command from the project root directory:

```bash
bash apply_model_adaptations.sh
```

The script copies the adapted conversation, model-building, language-model, and multimodal resampler implementations into `LLaVA-NeXT/llava/`. Existing files with matching names will be overwritten. Make sure `LLaVA-NeXT/` and `LLaVA-NeXT_replace_files/` are located in the project root before running the script.

### Step 4. Run Inference

We provide `infer_llava_gpu.py` for single-image inference with a trained VLM checkpoint. The script supports configurable model paths, conversation templates, and generation settings.

Run inference using the following command:

```bash
python infer_llava_gpu.py \
    --model-path /path/to/model/checkpoint \
    --model-name <model_name> \
    --image /path/to/image.jpg \
    --conv-template <conv_template> \
    --question "What is shown in this image?" \
    --device-map auto \
    --max-new-tokens 512
```

Replace `/path/to/model/checkpoint` with the path to your trained model checkpoint and `/path/to/image.jpg` with the path to your input image. Set `<model_name>` and `<conv_template>` according to the language model backbone.

The script loads the trained model, preprocesses the input image, and generates a textual response. GPU inference is required, and `--device-map auto` enables automatic device placement.

## Citation

If you find this work useful, please cite the following paper:

```bibtex
@misc{li2026transferstextvisioncapability,
    title={What Transfers from Text to Vision? Capability Scaling Laws and Transfer Dynamics for VLMs},
    author={Ziran Li and Qiang Wang and Zhengyu Chen and Shanglin Lei and Borun Chen and Jingang Wang and Xunliang Cai},
    year={2026},
    eprint={2608.00013},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2608.00013},
}
```
