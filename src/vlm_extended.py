from src.utils.prompts import EXTRACT_OBJECTS_TEMPLATE, PERSPECTIVE_CHANGE_TEMPLATE, EGOCENTRIC_REPHRASING_TEMPATE
from src.qwen_wrapper import QwenWrapper
from src.vision_external.external_vision_model import ExternalVisionModel

class VLMExtended:
    """
    Extended Visual Language Model (VLM) that incorporates an external vision model.
    """

    def __init__(self, vlm_path: str, external_vision_model, device: str):
        """
        Args:
            vlm_model: The base visual language model instance.
            external_vision_model (ExternalVisionModel, optional): An external vision model instance.
        """
        self.vlm_path = vlm_path
        self.vlm_model = QwenWrapper(vlm_path)
        self.vlm_model.load()
        
        self.external_vision_model = external_vision_model or ExternalVisionModel()
        self.current_task = None
        self.conversation = []

    def extract_objects_from_question(self, question):
        """
        Extract objects mentioned in the question.
        Args:
            question (str): The input question.
        Returns:
            list: List of extracted objects (dummy implementation).
        """
        message = {
            "role": "user",
            "content": [
            {"type": "text", "text": EXTRACT_OBJECTS_TEMPLATE.format(question=question)},
        ],
        }
        return self.vlm_model.generate(messages=[message])

    def set_reference_from_question(self, question):
        """
        Set a reference in the model based on the question.
        Args:
            question (str): The input question.
        """
        pass
    def generate_perspective_prompt(self, question, vision_input=None):
        """
        Generate a prompt for perspective-taking based on the question and vision input.
        Args:
            question (str): The input question.
            vision_input: Optional vision input to be processed.
        Returns:
            str: The generated prompt.
        """
        pass
