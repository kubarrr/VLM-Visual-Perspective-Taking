import math
import gc
import torch

from PIL import Image
from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast


class DepthProModelWrapper:
    def __init__(self, model_id: str = "apple/DepthPro-hf", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None

    def load(self):
        self.processor = DepthProImageProcessorFast.from_pretrained(self.model_id)
        self.model = DepthProForDepthEstimation.from_pretrained(self.model_id).to(
            self.device
        )

    def unload(self):
        if self.model is not None:
            self.model.to("cpu")
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def estimate_depth(self, img: Image.Image):
        psize = self.model.config.patch_size
        min_ratio = min(self.model.config.scaled_images_ratios)
        min_size = math.ceil(psize / min_ratio)
        H, W = img.height, img.width

        if min(H, W) < min_size:
            scale = min_size / min(H, W)
            img_proc = img.resize(
                (math.ceil(W * scale), math.ceil(H * scale)), Image.BICUBIC
            )
        else:
            img_proc = img

        inputs = self.processor(images=img_proc, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        results = self.processor.post_process_depth_estimation(
            out, target_sizes=[(H, W)]
        )
        depth_map = results[0]["predicted_depth"].squeeze().cpu().numpy()
        return depth_map
