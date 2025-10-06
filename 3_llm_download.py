# model_download.py
from modelscope import snapshot_download
import os
import argparse
"""3. Download the model from the Hugging Face Hub using the `snapshot_download` function from the `modelscope` library.

You can simply change this snapshot_download with huggingface's transformers library, but the modelscope library is more powerful and can be used to download models from other sources as well.

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="The model name.")
    parser.add_argument("--cache_dir", type=str, help="The cache directory.")
    args = parser.parse_args()
    model_dir = snapshot_download(args.model, cache_dir=args.cache_dir, revision='master')
    print(f"Model downloaded to: {model_dir}")
    # model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='/root/autodl-tmp', revision='master')
