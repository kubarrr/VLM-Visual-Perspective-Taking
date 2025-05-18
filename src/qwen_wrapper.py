import gc
import torch
from typing import Optional

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.utils.logger import setup_logger


class QwenWrapper:
    def __init__(self, model_name: str, log_file: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model: Optional[Qwen2_5_VLForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor] = None
        self.logger = setup_logger(__name__, log_file)

    def load(self):
        """Load model and tokenizer to the specified device."""
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.model.to(self.device)
        self.logger.info(f"Model loaded to {self.device}.")

    def unload(self):
        """Move model to CPU and free GPU memory."""
        if self.model is not None:
            self.model.to("cpu")
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("Model unloaded from GPU.")

    def generate(self, messages: list, **gen_kwargs):
        """Generate text from a prompt."""
        if self.model is None or self.processor is None:
            self.logger.error("Model not loaded. Call load() first.")

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text
