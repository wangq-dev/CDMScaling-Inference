# CDMScaling: VLM Inference

**EMNLP 2026 Main Conference (Oral)**

This repository provides model adaptations and a single-image GPU inference script for vision-language models (VLMs) trained using the [LLaVA-OneVision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/) framework. It extends the official [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) codebase to support different language model backbones used in our study.

### 🔗 Resources

- 📄 **Paper:** [What Transfers from Text to Vision? Capability Scaling Laws and Transfer Dynamics for VLMs](https://arxiv.org/abs/2608.00013)
- 🌐 **Project Website:** [CDMScaling](https://wangq-dev.github.io/CDMScaling-Website/)
- 💻 **Main Repository:** [CDMScaling](https://github.com/wangq-dev/CDMScaling)
- 🤖 **Model Weights:** [ModelScope](https://modelscope.ai/profile/wangq2k)

## Introduction

Our trained VLMs use [SigLIP-SO400M-Patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) as the vision encoder and are trained on selected portions of [Infinity-MM](https://huggingface.co/datasets/BAAI/Infinity-MM). Refer to the corresponding model card for backbone-specific configuration and training data details.

This repository focuses on **running inference with trained VLM checkpoints**. For capability scaling law fitting, and evaluation results, see the [main repository](https://github.com/wangq-dev/CDMScaling).

## How to Use

### Step 1. Set Up the Environment

Clone this repository and run the following commands from its root directory. Create a conda environment and install the required dependencies:

```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install -r requirements.txt
```

GPU inference requires a CUDA-compatible environment with the appropriate PyTorch and CUDA dependencies installed.

### Step 2. Install LLaVA-NeXT

Clone the official LLaVA-NeXT repository into the root directory of this project and install it in editable mode:

```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install -e .
cd ..
```

**Compatibility note:** The adaptation files may depend on the upstream LLaVA-NeXT version. If you have validated them against a specific commit, use that commit instead of relying on the latest default branch.

### Step 3. Apply Model Adaptations

Replace the corresponding files in LLaVA-NeXT with the modified implementations provided in this repository. From the project root, run:

```bash
bash apply_model_adaptations.sh
```

The script copies adapted conversation, model-building, and language-model into `LLaVA-NeXT/llava/`. **Files with matching names will be overwritten.** Ensure that both `LLaVA-NeXT/` and `LLaVA-NeXT_replace_files/` are located in the project root before running the script.

### Step 4. Run Inference

Download a compatible trained VLM checkpoint from [ModelScope](https://modelscope.ai/profile/wangq2k), or use an existing local checkpoint. The `infer_llava_gpu.py` script supports single-image inference with configurable model paths, conversation templates, and generation settings.

From the project root, run:

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

Replace `/path/to/model/checkpoint` with the checkpoint directory and `/path/to/image.jpg` with the input image path. Set `<model_name>` and `<conv_template>` to the values required by the selected language model backbone and its corresponding model implementation.

The script loads the model, preprocesses the image, and generates a textual response. GPU inference is required. The `--device-map auto` option enables automatic device placement, subject to available GPU memory and the installed dependencies.

## Citation

If you find this work useful, please cite our paper:

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
