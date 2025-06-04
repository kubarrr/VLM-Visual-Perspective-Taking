
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
        depth_map, focal_length = self.depthpro.estimate_depth(img)
        self.depthpro.unload()
        #4. Orient anything
        self.orient.load()
        orientations = self.orient.estimate_orientation(
            img, boxes
        )
        self.orient.unload()

        
        # 5) Compute median positions

        positions = []
        masks_np = masks.cpu().numpy()
        
        for m in masks_np:
            ys, xs = np.nonzero(m)

            def image_to_camera_coords(x, y, z, f, w, h):
                x_centered = x - (w / 2)
                y_centered = -y + (h / 2)
  
                X = (x_centered * z) / f
                Y = (y_centered * z) / f

                return X, Y

            w, h = img.size
            f = focal_length
            x_pixel, y_pixel = np.median(xs), np.median(ys)
            z_depth = np.median(depth_map[m.astype(bool)])

            X_cam, Y_cam = image_to_camera_coords(x_pixel, y_pixel, z_depth, f, w, h)
            positions.append((
                float(Y_cam),
                float(X_cam),
                float(z_depth)
            ))
            
        return {
            "positions": np.asarray(positions), # [N x (y, x, z)]
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "orientations": orientations,
        } 
