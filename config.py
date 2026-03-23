"""
Project configuration.
LLM settings use environment variable DASHSCOPE_API_KEY.
"""
import os
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = field(default_factory=lambda: os.environ.get("DASHSCOPE_API_KEY", ""))
    vlm_model: str = "qwen-vl-max"
    llm_model: str = "qwen-max"
    timeout: int = 30
    max_retries: int = 2


@dataclass
class MazeConfig:
    rows: int = 11
    cols: int = 11
    cell_size: int = 48


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    maze: MazeConfig = field(default_factory=MazeConfig)
    screenshot_dir: str = "out/screenshots"
    window_title: str = "CogNav Maze"
    fps: int = 30
    auto_step_interval: float = 0.3   # seconds between auto-nav steps


config = AppConfig()
