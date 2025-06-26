import argparse
import torch
from src.utils.constants import PERSPECTIVE_TYPE
from src.qwen_extended import QwenExtended
from src.utils.constants import QWEN_MODEL
from src.vision_module.external_vision_model import ExternalVisionModule
from huggingface_hub import login
from huggingface_hub.file_download import build_hf_headers
from datasets import load_dataset
import json


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run visual perspective taking pipeline."
    )
    parser.add_argument(
        "--output_folder", type=str, default="output/", help="Path to save the output."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Login using e.g. `huggingface-cli login` to access this dataset

    headers = build_hf_headers()  # handles authentication
    login("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")
    ds = load_dataset("Gracjan/Isle", "Isle-Brick-V1")

    output_folder = args.output_folder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    external_vision_m = ExternalVisionModule(device=device)
    vlm_extended = QwenExtended(
        vlm_path=QWEN_MODEL,
        external_vision_module=external_vision_m,
        renderer_module=None,
        output_folder=output_folder,
        device=device,
    )

    results = {}
    for idx, record in enumerate(ds["test"]):
        try:
            image, prompt, label = record["image"], record["prompt"], record["label"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            prompt = prompt.lower()
            a = vlm_extended.ask_question_with_perspective(
                prompt,
                image,
                perspective_type=PERSPECTIVE_TYPE.NUMERICAL,
                save_intermediate_name=f"{idx}_{prompt}",
            )

            a = a.lower()
            if a == "yes":
                a = 1
            elif a == "no":
                a = 0

            results[idx] = {"prompt": prompt, "label": label, "answer": a}

        except Exception as e:
            error_message = f"Error processing record {idx}: {str(e)}"
            with open(f"{output_folder}/errors.log", "a") as error_file:
                error_file.write(error_message + "\n")

    ## 2. count stats here
    correct_count = sum(1 for res in results.values() if res["answer"] == res["label"])
    total_count = len(results)
    accuracy = correct_count / total_count

    # 3. save to files
    with open(f"{output_folder}/stats.txt", "w") as stats_file:
        stats_file.write(f"Correct count: {correct_count}\n")
        stats_file.write(f"Total count: {total_count}\n")
        stats_file.write(f"Accuracy: {accuracy}\n")

    output_path = f"{output_folder}/results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
