from pathlib import Path

# pointing to `visual-perspective-taking-project`
ROOT_PATH = (Path(__file__) / ".." / ".." / "..") .resolve()
DATA_PATH = ROOT_PATH / "data"

QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
