"""Validated user-facing configuration, independent of any model family."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class AIConfig(BaseModel):
    base_url: str
    api_key: str | None = None
    model: str
    role_models: dict[str, str] = Field(default_factory=dict)


class DataConfig(BaseModel):
    root: Path
    heldout_split: str = "heldout"
    signals: list[str] = Field(default_factory=list)
    label_fields: list[str] = Field(default_factory=list)


class SearchConfig(BaseModel):
    providers: list[str] = Field(default_factory=lambda: ["arxiv", "crossref", "semantic_scholar", "github"])
    endpoint: str | None = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    adapter: str
    options: dict[str, Any] = Field(default_factory=dict)


class ProjectManifest(BaseModel):
    project_id: str
    objective: str
    constraints: list[str] = Field(default_factory=list)
    budget: float = 1.0
    model: ModelConfig
    data: DataConfig
    ai: AIConfig | None = None
    search: SearchConfig = Field(default_factory=SearchConfig)


def load_manifest(path: Path) -> ProjectManifest:
    import json
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProjectManifest.model_validate(value)
