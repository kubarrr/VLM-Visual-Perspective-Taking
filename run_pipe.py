import argparse
import torch
from PIL import Image
from src.utils.constants import PERSPECTIVE_TYPE
from src.qwen_extended import QwenExtended
from src.utils.constants import QWEN_MODEL
from src.vision_module.external_vision_model import ExternalVisionModule
from huggingface_hub import login
from huggingface_hub.file_download import build_hf_headers
import requests
from mlcroissant import Dataset
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run visual perspective taking pipeline.")
    parser.add_argument("--output_folder", type=str, default="output/", help="Path to save the output.")
    return parser.parse_args()



if __name__ == "__main__":
    # Login using e.g. `huggingface-cli login` to access this dataset
    headers = build_hf_headers()  # handles authentication
    login("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")
    jsonld = requests.get("https://huggingface.co/api/datasets/Gracjan/Isle/croissant", headers=headers).json()
    ds = Dataset(jsonld=jsonld)

    data_list = []

    args = parse_arguments()
    output_folder = args.output_folder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    external_vision_m = ExternalVisionModule(device=device)
    vlm_extended = QwenExtended(vlm_path=QWEN_MODEL, 
                                external_vision_module=external_vision_m, 
                                renderer_module=None, 
                                output_folder=output_folder,
                                device=device)
    count = 0
    limit = 100


    for record in ds.records("Isle-Brick-V1"):
        if count >= limit:  
            break
        image, prompt, label = record['Isle-Brick-V1/image'], record['Isle-Brick-V1/prompt'], record['Isle-Brick-V1/label']

        if image.mode != "RGB":
            image = image.convert("RGB")

        a = vlm_extended.ask_question_with_perspective(prompt, image, perspective_type=PERSPECTIVE_TYPE.NUMERICAL, save_intermediate=True)
        print(f"Image label: {label}, Prmpt: {prompt}, Answer: {a}")
        count += 1

    
    

    