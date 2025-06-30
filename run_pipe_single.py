import argparse
import torch
from PIL import Image
from src.utils.constants import PERSPECTIVE_TYPE
from src.qwen_extended import QwenExtended
from src.utils.constants import QWEN_MODEL
from src.vision_module.external_vision_model import ExternalVisionModule

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run visual perspective taking pipeline.")
    parser.add_argument("--image_path", type=str, default="data/pixelized.png", help="Path to the input image.")
    parser.add_argument("--question", type=str, default="From the woman's perspective is the dog on left or right of the chair?", help="Path to the input image.")
    parser.add_argument("--output_folder", type=str, default="output/", help="Path to save the output.")
    return parser.parse_args()



if __name__ == "__main__":
    
    args = parse_arguments()
    output_folder = args.output_folder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load models
    external_vision_m = ExternalVisionModule(device=device)
    vlm_extended = QwenExtended(vlm_path=QWEN_MODEL, 
                                external_vision_module=external_vision_m, 
                                renderer_module=None, 
                                output_folder=output_folder,
                                device=device)
    
    # prepare data
    q = args.question
    img_path = args.image_path
    img = Image.open(img_path).convert("RGB")
    
    # run
    a = vlm_extended.ask_question_with_perspective(q, img, perspective_type=PERSPECTIVE_TYPE.NUMERICAL, save_intermediate_name="intermediate")
    

    