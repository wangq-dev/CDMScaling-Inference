
import argparse
import copy
import warnings

import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLaVA inference on GPU."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="llava_qwen3",
        help="Model name (e.g., llava_qwen3)",
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image",
    )

    parser.add_argument(
        "--question",
        type=str,
        default="What is shown in this image?",
        help="Question to ask about the image",
    )

    parser.add_argument(
        "--conv-template",
        type=str,
        default="llava_qwen3",
        help="Name of the conversation template",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )

    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device mapping strategy (default: auto)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. GPU inference cannot proceed."
        )

    print("=" * 80)
    print("[GPU inference configuration]")
    print("=" * 80)
    print(f"Model path    : {args.model_path}")
    print(f"Model name    : {args.model_name}")
    print(f"Image         : {args.image}")
    print(f"Conv template : {args.conv_template}")
    print(f"Device map    : {args.device_map}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # Load the pretrained model
    tokenizer, model, image_processor, context_len = (
        load_pretrained_model(
            args.model_path,
            None,
            args.model_name,
            device_map=args.device_map,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
        )
    )

    model.eval()

    # Determine the input device and model data type
    try:
        input_device = model.get_input_embeddings().weight.device
    except Exception:
        input_device = torch.device("cuda:0")

    model_dtype = model.dtype

    print(f"Input device  : {input_device}")
    print(f"Model dtype   : {model_dtype}")
    print(f"Context length: {context_len}")

    # Load the input image
    image = Image.open(args.image).convert("RGB")
    image_sizes = [image.size]

    print(f"Image size    : {image.size}")

    # Preprocess the image
    image_tensor = process_images(
        [image],
        image_processor,
        model.config,
    )

    image_tensor = [
        item.to(
            device=input_device,
            dtype=model_dtype,
        )
        for item in image_tensor
    ]

    # Construct the conversation prompt
    if args.conv_template not in conv_templates:
        available_templates = sorted(
            conv_templates.keys()
        )

        raise KeyError(
            f"Conversation template not found: {args.conv_template}\n"
            f"Available templates: {available_templates}"
        )

    question = (
        DEFAULT_IMAGE_TOKEN
        + "\n"
        + args.question
    )

    conv = copy.deepcopy(
        conv_templates[args.conv_template]
    )

    conv.append_message(
        conv.roles[0],
        question,
    )

    conv.append_message(
        conv.roles[1],
        None,
    )

    prompt = conv.get_prompt()

    print("\n[Prompt]")
    print(prompt)

    # Tokenize the input prompt
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )

    input_ids = input_ids.unsqueeze(0).to(
        input_device
    )

    # Define the stopping criteria
    stopping_criteria = KeywordsStoppingCriteria(
        keywords=[conv.sep],
        tokenizer=tokenizer,
        input_ids=input_ids,
    )

    # Generate the model response
    print("\n[Generating...]")

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=args.max_new_tokens,
            stopping_criteria=[
                stopping_criteria
            ],
            use_cache=True,
        )

    # Decode the generated token IDs
    output_text = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=False,
    )

    print("\n[Output]")
    for text in output_text:
        print(text.strip())


if __name__ == "__main__":
    main()
