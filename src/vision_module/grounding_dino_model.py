import gc
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = (0.3,)


class GroundingDINOModelWrapper:
    def __init__(
        self, model_id="IDEA-Research/grounding-dino-tiny", device: str = "cuda"
    ):
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)

    def unload(self):
        if self.model is not None:
            self.model.to("cpu")
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def detect(self, img, objects: list):
        
        # dino expects input in str in format: obj1. obj2 obj3.
        text_dino = ". ".join(objects) + "."
        
        inputs = self.processor(images=img, text=text_dino, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        detection_results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[img.size[::-1]],
        )[0]
        return detection_results["boxes"].cpu().numpy().tolist(), detection_results["labels"]
