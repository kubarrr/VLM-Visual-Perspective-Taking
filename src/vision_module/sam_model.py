import gc
from typing import Optional

import torch
from transformers import SamModel, SamProcessor


class SAMModelWrapper:
    def __init__(self, model_id="facebook/sam-vit-base", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None

    def load(self):
        self.processor = SamProcessor.from_pretrained(self.model_id)
        self.model = SamModel.from_pretrained(self.model_id).to(self.device)

    def unload(self):
        if self.model is not None:
            self.model.to("cpu")
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_masks(self, img, boxes):
        sam_inputs = self.processor(
            images=img, input_boxes=[boxes], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            sam_outputs = self.model(**sam_inputs)
        masks = self.processor.post_process_masks(
            sam_outputs.pred_masks,
            original_sizes=sam_inputs.original_sizes,
            reshaped_input_sizes=sam_inputs.reshaped_input_sizes,
        )[0]
        # SAM returns many masks for each object, we return mask with higest IoU
        return masks[:, 0, :, :] 
