#!/bin/bash
set -e

cp LLaVA-NeXT_replace_files/llava/conversation.py \
   LLaVA-NeXT/llava/conversation.py

cp LLaVA-NeXT_replace_files/llava/model/__init__.py \
   LLaVA-NeXT/llava/model/__init__.py

cp LLaVA-NeXT_replace_files/llava/model/builder.py \
   LLaVA-NeXT/llava/model/builder.py

cp LLaVA-NeXT_replace_files/llava/model/multimodal_encoder/builder.py \
   LLaVA-NeXT/llava/model/multimodal_encoder/builder.py

cp -r LLaVA-NeXT_replace_files/llava/model/language_model/. \
   LLaVA-NeXT/llava/model/language_model/

cp -r LLaVA-NeXT_replace_files/llava/model/multimodal_resampler/. \
   LLaVA-NeXT/llava/model/multimodal_resampler/

echo "Model adaptations applied successfully."
