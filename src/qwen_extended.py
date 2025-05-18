"""
Extended Visual Language Model (VLM) that integrates an external vision model 
and renderer for advanced spatial reasoning tasks. 
It's main task is to answer spatial reasoning and vpt questions.
"""

from src.utils.prompts import (
    EXTRACT_OBJECTS_TEMPLATE,
    PERSPECTIVE_CHANGE_TEMPLATE,
    EGOCENTRIC_REPHRASING_TEMPATE,
)
from src.qwen_wrapper import QwenWrapper
from src.vision_module.external_vision_model import ExternalVisionModule
from src.render_module.renderer_module import Renderer
from src.vlm_extended import VLMExtended

from src.utils.constants import PERSPECTIVE_TYPE
from src.utils.logger import setup_logger

class QwenExtended(VLMExtended):
    """
    Extended Visual Language Model (VLM) that incorporates an external vision model.
    """

    def __init__(
        self,
        vlm_path: str,
        external_vision_module: ExternalVisionModule,
        renderer_module: Renderer,
        device: str,
        log_file: str
    ):
        """
        Args:
            vlm_model: The base visual language model instance.
            external_vision_model (ExternalVisionModel, optional): An external vision model instance.
        """
        self.device = device
        self.logger = setup_logger(__name__, log_file)
            
        
        self.vlm_path = vlm_path
        self.vlm_model = QwenWrapper(vlm_path, device=self.device, log_file=log_file)
        self.vlm_model.load()

        # hold external modules
        self.external_vision_model = external_vision_module or ExternalVisionModule()
        self.renderer_module = renderer_module or Renderer()
        self.current_task = None
        self.conversation = []

    
    def ask_question_with_perspective(self, 
                                      question: str, 
                                      perspective_type: PERSPECTIVE_TYPE) -> str:
        """
        All workflow to generate answer for spatial reasoning quesiton with external tools 
        Args:
            question (str): The input question.
            perspective_type 
        Returns:
            str: vlm answer.
        """        
        # 1. get objects in interest
        objects = self.extract_objects_from_question(question)
        
        # 2. process with external module
        scene = self.external_vision_model.abstract_scene(self.current_task, objects)
        
        # 3. convert question to egocentric
        egocentric_question = self.rephrase_to_egocentric(question)
        
        # 4. generate perspective prompt
        perspective_prompt = self.generate_perspective_prompt(egocentric_question=egocentric_question,
                                                              scene_abstraction=scene,
                                                              perspective_type=perspective_type)
        
        # 5. ask final question with auxilary perspective prompt
        prompt = question + perspective_prompt
        return self.vlm_model.generate(messages=[prompt])
    
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
                {
                    "type": "text",
                    "text": EXTRACT_OBJECTS_TEMPLATE.format(question=question),
                },
            ],
        }
        return self.vlm_model.generate(messages=[message])

    def rephrase_to_egocentric(self, question) -> str:
        """
        Rephrase the question to an egocentric perspective.
        Args:
            question (str): The input question.
        """
        pass

    def generate_perspective_prompt(self, 
                                    egocentric_question: str, 
                                    scene_abstraction: list,
                                    perspective_type: PERSPECTIVE_TYPE):
        """
        Generate a prompt for perspective-taking based on the question visual or numerical prompt.
        Args:
            question (str): The input question.
            perspective_type: Numerical or visual
        Returns:
            str: The generated auxilary prompt.
        """
        pass
    
    
