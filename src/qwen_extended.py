"""
Extended Visual Language Model (VLM) that integrates an external vision model
and renderer for advanced spatial reasoning tasks.
It's main task is to answer spatial reasoning and vpt questions.
"""
import os
from typing import Optional
from PIL import Image
from src.utils.prompts import (
    EXTRACT_OBJECTS_TEMPLATE,
    PERSPECTIVE_CHANGE_TEMPLATE,
    EGOCENTRIC_REPHRASING_TEMPLATE,
    PERSPECTIVE_PROMPT_TEMPLATE,
)
from src.qwen_wrapper import QwenWrapper
from src.vision_module.external_vision_model import ExternalVisionModule
from src.render_module.renderer_module import Renderer
from src.vlm_extended import VLMExtended

from src.utils.constants import PERSPECTIVE_TYPE
from src.utils.logger import setup_logger
from src.utils.utils import (
    llm_output_to_list,
    get_labels_positions_without_central,
    change_points_basis,
)


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
        output_folder: str,
    ):
        """
        Args:
            vlm_model: The base visual language model instance.
            external_vision_model (ExternalVisionModel, optional): An external vision model instance.
        """
        self.device = device
        
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        
        log_file = os.path.join(self.output_folder, "project.log")
        self.logger = setup_logger(__name__, log_file)

        self.vlm_path = vlm_path
        self.vlm_model = QwenWrapper(vlm_path, device=self.device, log_file=log_file)
        self.vlm_model.load()

        # hold external modules
        self.external_vision_model = external_vision_module or ExternalVisionModule()
        self.renderer_module = renderer_module or Renderer()

    def ask_question_with_perspective(
        self, 
        question: str, 
        img: Image, 
        perspective_type: PERSPECTIVE_TYPE,
        save_intermediate: bool,
    ) -> str:
        """
        All workflow to generate answer for spatial reasoning quesiton with external tools
        Args:
            question (str): The input question.
            img (PIL.Image): Input image
            perspective_type: NUMERICAL / VISUAL
        Returns:
            str: vlm answer.
        """
        # 1. get objects in interest
        objects = self.extract_objects_from_question(question)
        self.logger.info(f"Objects extracted from question: {objects}")
        # 2. process with external module
        intermediate_save_path = os.path.join(self.output_folder, "intermediate.png") if save_intermediate else None
        scene = self.external_vision_model.abstract_scene(img=img, objects=objects, save_img_path=intermediate_save_path)

        self.logger.info("Labels dino: %s", scene["labels"])
        self.logger.info(f"Scene abstraction finished")

        # 3. convert question to egocentric
        egocentric_question = self.rephrase_to_egocentric(question)
        self.logger.info(f"Egocentric question: {egocentric_question}")

        # 4. extract central perspective
        central_perspective = self.find_perspective(question=question, options=objects)
        self.logger.info(f"Central perspective detected: {central_perspective}")
        
        
        # 5. generate perspective prompt
        perspective_prompt = self.generate_perspective_prompt(
            egocentric_question=egocentric_question,
            scene_abstraction=scene,
            central_perspective=central_perspective,
            perspective_type=perspective_type,
        )
        self.logger.info(f"Perspective prompt generated: {perspective_prompt}")

        # 5. ask final question with auxilary perspective prompt
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": perspective_prompt,
                },
            ],
        }
        # prompt = question + perspective_prompt
        # 6. get answer
        answer = self.vlm_model.generate(messages=[message])
        self.logger.info(f"Answer from VLM: {answer}")
        return answer

    def extract_objects_from_question(self, question):
        """
        Extract objects mentioned in the question.
        Args:
            question (str): The input question.
        Returns:
            list: List of extracted objects.
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
        answer = self.vlm_model.generate(messages=[message])[0]
        objects = llm_output_to_list(answer)
        if objects is None:
            self.logger.error("Failed to extract objects from question.")
            raise ValueError()
        return objects

    def find_perspective(self, question: str, options: list) -> str:
        """
        Find the perspective from which the question is asked.
        Args:
            question (str): The input question.
            options (list): List of objects extracted from scene.
        Returns:
            str: The detected perspective.
        """
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PERSPECTIVE_CHANGE_TEMPLATE.format(
                        question=question, options=options
                    ),
                },
            ],
        }
        answer = self.vlm_model.generate(messages=[message])[0]
        return answer

    def rephrase_to_egocentric(self, question) -> str:
        """
        Rephrase the question to an egocentric perspective.
        Args:
            question (str): The input question.
        """
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": EGOCENTRIC_REPHRASING_TEMPLATE.format(question=question),
                },
            ],
        }
        egocentric_question = self.vlm_model.generate(messages=[message])[0]
        return egocentric_question

    def generate_perspective_prompt(
        self,
        egocentric_question: str,
        scene_abstraction: dict,
        central_perspective: str,
        perspective_type: PERSPECTIVE_TYPE,
    ):
        """
        Generate a prompt for perspective-taking based on the question visual or numerical prompt.
        Args:
            question (str): The input question.
            perspective_type: Numerical or visual
        Returns:
            str: The generated auxilary prompt.
        """
        # get labels and positions WITHOUT central_perspective object
        labels_remaining, positions_remaining = get_labels_positions_without_central(
            results=scene_abstraction, central_perspective=central_perspective
        )
        # extract index for central perspective object
        index_central_perspective = scene_abstraction["labels"].index(
            central_perspective
        )
    
        translation, euler_angles = (
            scene_abstraction["positions"][index_central_perspective],
            scene_abstraction["orientations"][index_central_perspective, :3],
        )

        # change basis of remaining points using central perspective as base
        positions_egocentric_base = change_points_basis(
            euler_angles=euler_angles,
            translation=translation,
            points=positions_remaining,
        )

        # numerical -> we feed new coordinates directly to prompt
        if perspective_type == PERSPECTIVE_TYPE.NUMERICAL:
            # throughout the pipeline we keep 3d coordinates as [y, x, z]
            # however for vlm prompt we should change it to [x, y, z]
            # For now I change it here, but it might not be most elegant solution and I am happy to change it.
            coordinates_dict = {
                label: [row[1].round(2).item(), row[0].round(2).item(), row[2].round(2).item()] # [y, x, z] -> [x, y, z]
                for label, row in zip(labels_remaining, positions_egocentric_base)
            }

            return PERSPECTIVE_PROMPT_TEMPLATE.format(
                    source=central_perspective,
                    coordinates=coordinates_dict,
                    question=egocentric_question)
        
        # TODO visual -> we need to render the scene and then feed the rendered image to prompt
        else:
            raise ValueError("TO DO visual perspective.")
