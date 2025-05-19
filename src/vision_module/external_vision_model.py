"""
ExternalVisionModule is responsible for all scene abstraction flow. From raw image, to list of objects' poses (position + orientation)
"""
class ExternalVisionModule:
    """
    Stub for an external vision model.
    As we will, use many big models, which consumes a lot of memory, please make sure we can load / unload models dynamically 
    before and after inference. (example approach in src/qwen_wrapper.py)
    """

    def __init__(self, model_name: str = "GenericVisionModel"):
        pass
    def abstract_scene(self, img, objects) -> list:
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

        # TODO 
        pass