
"""
ExternalVisionModule is responsible for all scene abstraction flow. From raw image, to list of objects' poses (position + orientation)
"""
import numpy as np
from PIL import Image
from typing import List

from .grounding_dino_model import GroundingDINOModelWrapper
from .sam_model import SAMModelWrapper
from .depthpro_model import DepthProModelWrapper
from .orient_anything_model import OrientAnythingModelWrapper

class ExternalVisionModule:
    """
    Refactored external vision model using modular model classes.
    """

    def __init__(self, model_name: str = "GenericVisionModel", device: str = "cuda"):
        self.device = device  # Let model classes handle device selection
        self.dino = GroundingDINOModelWrapper(device=self.device)
        self.sam = SAMModelWrapper(device=self.device)
        self.depthpro = DepthProModelWrapper(device=self.device)
        self.orient = OrientAnythingModelWrapper(device=device)

    def abstract_scene(self, img: Image.Image, objects: List[str]) -> list:
        """
        ExternalVisionModule exposes single method for VLMExtended
        `abstract_scene` gets image and list of objects (e.g [woman, dog, chair])
        it processes input using all models sequentially (grounding dino, sam, ...)
        and returns list of positions and orientations of each object.
        Args:
            img ([type]): [description]
            objects ([type]): [description]
        Returns:
            list: list with coordinates and orientations of each object, so vlm can
                  later transform it into numerical or visual prompt (look paper)
        """


        # 1. Object detection with GroundingDINO
        self.dino.load()
        boxes, labels = self.dino.detect(img, objects)
        self.dino.unload()
        # 2. SAM masks
        self.sam.load()
        masks = self.sam.get_masks(img, boxes)
        self.sam.unload()
        # 3. Depth estimation
        self.depthpro.load()
        depth_map = self.depthpro.estimate_depth(img)
        self.depthpro.unload()
        #4. Orient anything
        self.orient.load()
        orientations = self.orient.estimate_orientation(
            img, boxes, masks=masks, depth_map=depth_map
        )
        self.orient.unload()

        
        # 5) Compute median positions

        positions = []
        masks_np = masks.cpu().numpy()
        
        masks_number, height, width = masks_np.shape
        for mask_idx in range(masks_number):
            xs, ys = np.nonzero(masks_np[mask_idx])
            x_med, y_med = np.median(xs), np.median(ys)
            
            z_med = np.median(depth_map[masks_np[mask_idx]])   
            
            positions.append((float(x_med), float(y_med), float(z_med)))
            
        return {
            "positions": np.asarray(positions), # [N x (y, x, z)]
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "orientations": orientations,
        } 
