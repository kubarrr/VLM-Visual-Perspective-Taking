
import os
import gc
import torch
from PIL import Image
from transformers import AutoImageProcessor
from huggingface_hub import hf_hub_download
from .orient_anything.vision_tower import DINOv2_MLP
from .orient_anything.inference import get_3angle
import numpy as np
from scipy.spatial.transform import Rotation


class OrientAnythingModelWrapper:
    _OUT_DIM = 360 + 180 + 180 + 2

    def __init__(self, device="cuda", cache_dir=None):
        self.device = device
        self.cache_dir = cache_dir or os.getcwd()
        self.repo_id = "Viglong/Orient-Anything"
        self.dino_mlp = None
        self.processor = None

    def load(self):
        weight_path = hf_hub_download(
            repo_id=self.repo_id,
            filename="croplargeEX2/dino_weight.pt",
            cache_dir=self.cache_dir,
            repo_type="model",
        )
        self.dino_mlp = DINOv2_MLP(
            dino_mode="large",
            in_dim=1024,
            out_dim=self._OUT_DIM,
            evaluate=True,
            mask_dino=False,
            frozen_back=False,
        )
        state = torch.load(weight_path, map_location="cpu")
        self.dino_mlp.load_state_dict(state)
        self.dino_mlp = self.dino_mlp.to(self.device).eval()

        backbone_id = "facebook/dinov2-large"
        self.processor = AutoImageProcessor.from_pretrained(
            backbone_id, cache_dir=self.cache_dir
        )

    def unload(self):
        if self.dino_mlp is not None:
            self.dino_mlp.to("cpu")
            del self.dino_mlp, self.processor
            self.dino_mlp = None
            self.processor = None
            torch.cuda.empty_cache()
            gc.collect()

    def estimate_orientation(self, img: Image.Image, boxes):
        results = []
        for box in boxes:
            x0, y0, x1, y1 = map(int, box)
            crop = img.crop((x0, y0, x1, y1)).convert("RGB")
            angles = get_3angle(crop, self.dino_mlp, self.processor, self.device)
            # angles[0] = angles[0] + 180 if angles[0] < 180 else angles[0] - 180

            results.append(tuple(angles.cpu().tolist()))
        return np.array(results)
    
    def estimate_orientation_just_image(self, img: Image.Image):
        print(img)
        angles = get_3angle(img, self.dino_mlp, self.processor, self.device)
        # angles = [90, 0, 0]
        angles[0] = angles[0] - 180
        angles[1] = angles[1]
        r = Rotation.from_euler("yxz", angles=angles[:3], degrees=True).inv()
        rotation_matrix = r.as_matrix()
        new_positions = [0, 0, -1] @ rotation_matrix.T
        return np.array(angles), new_positions
