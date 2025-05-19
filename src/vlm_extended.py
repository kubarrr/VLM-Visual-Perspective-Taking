from typing import List
from abc import ABC, abstractmethod
from src.utils.constants import PERSPECTIVE_TYPE


class VLMExtended(ABC):
    """
    Abstract base class for Extended Visual Language Model (VLM).
    Defines the interface for VLMs that integrate external vision and rendering modules.
    """
    

    @abstractmethod
    def ask_question_with_perspective(
        self, question: str, perspective_type: PERSPECTIVE_TYPE
    ) -> str:
        """
        Full workflow to generate an answer using external tools.
        Args:
            question (str): The input question.
            perspective_type (PERSPECTIVE_TYPE): The type of perspective (numerical or visual).
        Returns:
            str: The VLM answer.
        """
        pass

    @abstractmethod
    def extract_objects_from_question(self, question: str) -> List[str]:
        """
        Extract objects mentioned in the question.
        Args:
            question (str): The input question.
        Returns:
            list: List of extracted objects.
        """
        pass

    @abstractmethod
    def rephrase_to_egocentric(self, question: str) -> str:
        """
        Rephrase the question to an egocentric perspective.
        Args:
            question (str): The input question.
        Returns:
            str: The egocentric version of the question.
        """
        pass

    @abstractmethod
    def generate_perspective_prompt(
        self,
        egocentric_question: str,
        scene_abstraction: list,
        perspective_type: PERSPECTIVE_TYPE,
    ) -> str:
        """
        Generate a prompt for perspective-taking based on the question and scene abstraction.
        Args:
            egocentric_question (str): The egocentric version of the question.
            scene_abstraction (list): List of objects with positions and orientations.
            perspective_type (PERSPECTIVE_TYPE): Numerical or visual.
        Returns:
            str: The generated auxiliary prompt.
        """
        pass
