import argparse
import torch
from PIL import Image
from src.utils.constants import PERSPECTIVE_TYPE
from src.qwen_extended import QwenExtended
from src.utils.constants import QWEN_MODEL
from src.vision_module.external_vision_model import ExternalVisionModule
from src.vision_module.orient_anything_model import OrientAnythingModelWrapper

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run orient anything.")
    parser.add_argument("--image_path", type=str, default="data/pixelized.png", help="Path to the input image.")
    return parser.parse_args()



if __name__ == "__main__":
    
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_path = args.image_path
    img = Image.open(img_path).convert("RGB")

    orient = OrientAnythingModelWrapper(device=device)
    
    # run
    orient.load()
    print(orient.estimate_orientation_just_image(img))
    

    