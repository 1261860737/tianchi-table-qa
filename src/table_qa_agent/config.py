"""项目配置读取与环境变量覆盖。"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover - 仅兼容未重新安装依赖的开发环境
    _load_dotenv = None


class ProviderConfig(BaseModel):
    """OpenAI 兼容视觉模型配置。"""

    api_key_env: str = "DASHSCOPE_API_KEY"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3.7-plus"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=256)
    timeout_seconds: float = Field(default=180.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    json_mode: bool = True
    enable_thinking: bool = False
    min_pixels: int | None = Field(default=262_144, gt=0)
    max_pixels: int | None = Field(default=8_388_608, gt=0)

    @model_validator(mode="after")
    def validate_pixel_range(self) -> ProviderConfig:
        if (
            self.min_pixels is not None
            and self.max_pixels is not None
            and self.min_pixels > self.max_pixels
        ):
            raise ValueError("min_pixels 不能大于 max_pixels")
        return self


class DocumentConfig(BaseModel):
    """文档转图与请求体限制。"""

    cache_dir: Path = Path("outputs/cache/documents")
    pdf_dpi: int = Field(default=220, ge=72, le=600)
    max_pages: int = Field(default=12, ge=1)
    max_long_edge: int = Field(default=4096, ge=512)
    jpeg_quality: int = Field(default=92, ge=50, le=100)
    max_encoded_bytes: int = Field(default=18 * 1024 * 1024, ge=1024 * 1024)


class RuntimeConfig(BaseModel):
    """批处理和日志配置。"""

    log_dir: Path = Path("outputs/logs")
    continue_on_error: bool = True
    max_workers: int = Field(default=8, ge=1, le=32)
    request_interval_seconds: float = Field(default=0.0, ge=0.0)


class PlanningConfig(BaseModel):
    """模型意图规划；失败时由本地规则 Planner 确定性兜底。"""

    model_intent_enabled: bool = True
    fallback_to_rules: bool = True


class RetrievalConfig(BaseModel):
    """首次失败后的页面定位和高清区域裁剪。"""

    enabled: bool = True
    roi_dpi: int = Field(default=360, ge=150, le=600)
    padding_ratio: float = Field(default=0.03, ge=0.0, le=0.2)
    max_regions: int = Field(default=3, ge=1, le=8)


class OCRConfig(BaseModel):
    """可插拔 OCR 候选生成；默认只在视觉恢复时调用。"""

    enabled: bool = True
    model: str | None = None
    max_prompt_chars: int = Field(default=6000, ge=500, le=20000)


class RecoveryConfig(BaseModel):
    """按错误层级进行有限次数恢复，避免无限 Agent 循环。"""

    enabled: bool = True
    max_operation_repairs: int = Field(default=1, ge=0, le=2)
    max_visual_retries: int = Field(default=1, ge=0, le=1)
    max_agent_tool_calls: int = Field(default=1, ge=0, le=3)
    repair_structure_dimensions: bool = True
    max_structure_patches: int = Field(default=1, ge=0, le=1)


class AppConfig(BaseModel):
    """应用完整配置。"""

    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    documents: DocumentConfig = Field(default_factory=DocumentConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)


def _load_project_dotenv(path: Path) -> None:
    """优先用 python-dotenv；缺失时解析本项目所需的简单 KEY=VALUE。"""

    if _load_dotenv is not None:
        _load_dotenv(dotenv_path=path, override=False)
        return
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def load_config(path: Path | str) -> AppConfig:
    """加载 YAML，并允许常用环境变量覆盖模型连接信息。"""

    config_path = Path(path)
    # 默认从当前项目目录加载 .env；真实环境变量优先，不会被覆盖。
    _load_project_dotenv(Path.cwd() / ".env")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    config = AppConfig.model_validate(raw)

    overrides: dict[str, object] = {}
    if base_url := os.getenv("TABLE_QA_BASE_URL") or os.getenv("BASE_URL"):
        overrides["base_url"] = base_url
    if model := os.getenv("TABLE_QA_MODEL"):
        overrides["model"] = model
    if overrides:
        config.provider = config.provider.model_copy(update=overrides)
    return config
