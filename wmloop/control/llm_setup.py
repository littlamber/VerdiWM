"""First-contact guidance and local inspection for the research LLM."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from wmloop.execute.configured_llm_broker import (
    ConfiguredBrokerError,
    default_config_path,
    load_config,
)


GUIDE = r"""VERDI 需要一个 LLM API 来完成研究助手工作：读取 IRG/诊断结果、跨域检索、
提出候选方法和生成受约束的 adapter 草案。它不是你的 Wan、JEPA 或其他世界模型
的 API；模型权重、数据和 GPU 运行时仍然在你自己的机器上。

推荐配置（API key 只放在本机权限为 600 的文件中）：

  mkdir -p CONFIG_DIR
  cat > CONFIG_PATH <<'EOF'
  [llm]
  base_url = "https://api.openai.com"
  model = "gpt-4.1"
  api_style = "responses"
  token_file = "auth"
  EOF
  read -r -s -p 'LLM API key: ' VERDI_LLM_KEY
  printf '\n'
  printf '%s' "$VERDI_LLM_KEY" > AUTH_PATH
  unset VERDI_LLM_KEY
  chmod 600 AUTH_PATH

然后检查配置（只读本地文件和环境变量，不会发网络请求，也不会显示 key）：

  verdi llm status

第三方 OpenAI-compatible 服务可以把 base_url、model 换成服务商的值。
如果服务只提供 Chat Completions，把 api_style 改为 "chat_completions"：

  [llm]
  base_url = "https://api.example.com"
  model = "provider-model"
  api_style = "chat_completions"
  token_environment_key = "MY_VERDI_LLM_TOKEN"

  read -r -s -p 'LLM API key: ' VERDI_LLM_KEY
  printf '\n'
  export MY_VERDI_LLM_TOKEN="$VERDI_LLM_KEY"
  unset VERDI_LLM_KEY

使用环境变量时不要再设置 token_file。配置文件也可以放在项目外的其他位置，
通过 VERDIWM_CONFIG=/path/to/config.toml 或命令行 --llm-config 指定。

本地 Ollama/vLLM 等无需认证的 OpenAI-compatible 服务可显式写：

  [llm]
  base_url = "http://127.0.0.1:11434"
  model = "qwen2.5:32b"
  api_style = "chat_completions"
  auth_required = false

只有 localhost、127.0.0.1 和 ::1 允许使用 HTTP；远程服务必须使用 HTTPS。

在研究流程中，配置已生效后可以把它传给需要研究 LLM 的命令，例如：

  verdi run --model /path/to/model --data /path/to/data \
    --goal "improve minute-scale consistency" --llm-config ~/.config/verdiwm/config.toml
  verdi repair-adapter ... --llm-config ~/.config/verdiwm/config.toml

不要把真实 key 写入 verdiwm.toml、研究计划、questionnaire、evidence、日志或 Git。
VERDI 只会记录 provider/model 和凭据来源，不会把 key 写入研究产物；检索结果仍须
经过目标侧 evaluator、冻结 verifier 和证据门禁，LLM 本身不能宣布实验成功。
"""


def _credential_status(config: dict[str, Any]) -> dict[str, Any]:
    required = bool(config.get("auth_required", True))
    token_file = config.get("token_file")
    env_key = str(config.get("token_environment_key", "VERDIWM_LLM_BROKER_TOKEN"))
    if not required:
        return {"source": "none", "configured": True, "required": False}
    if isinstance(token_file, Path):
        try:
            info = token_file.lstat()
        except OSError:
            return {
                "source": "token_file",
                "configured": False,
                "required": True,
                "path": str(token_file),
                "problem": "token_file_missing",
            }
        if token_file.is_symlink() or not stat.S_ISREG(info.st_mode):
            problem = "token_file_invalid"
        elif info.st_mode & 0o077:
            problem = "token_file_permissions"
        elif info.st_size > 16384:
            problem = "token_file_too_large"
        else:
            try:
                value = token_file.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                value = ""
            problem = None if value and "\x00" not in value and "\n" not in value and "\r" not in value else "token_file_empty_or_invalid"
        return {
            "source": "token_file",
            "configured": problem is None,
            "required": True,
            "path": str(token_file),
            "permissions": "owner_only" if not (info.st_mode & 0o077) else "not_owner_only",
            **({"problem": problem} if problem else {}),
        }
    configured = bool(os.environ.get(env_key, ""))
    return {
        "source": "environment",
        "key": env_key,
        "configured": configured,
        "required": True,
        **({} if configured else {"problem": "environment_variable_missing"}),
    }


def inspect_llm_config(path: Path | None = None) -> dict[str, Any]:
    """Inspect local LLM configuration without contacting the provider."""

    source = (path or default_config_path()).expanduser()
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "verdiwm-llm-configuration-status",
        "config_path": str(source),
    }
    if not source.exists():
        report.update({
            "state": "missing",
            "action": "运行 verdi llm 查看配置教程，然后运行 verdi llm status 复查。",
        })
        return report
    try:
        config = load_config(source)
    except ConfiguredBrokerError as exc:
        report.update({
            "state": "invalid",
            "error": str(exc).split(":", 1)[0],
            "action": "运行 verdi llm 查看示例，并修正配置文件后重试。",
        })
        return report
    token_key = str(config.get("token_environment_key", ""))
    if bool(config.get("auth_required", True)) and not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", token_key
    ):
        report.update({
            "state": "invalid",
            "error": "OPENAI_BROKER_TOKEN_KEY_INVALID",
            "action": "token_environment_key 必须是合法的环境变量名；请运行 verdi llm 查看示例。",
        })
        return report
    endpoint = str(config.get("endpoint") or config.get("base_url") or "")
    try:
        parsed = urlparse(endpoint)
    except ValueError:
        parsed = None
    valid_endpoint = parsed is not None and bool(parsed.hostname) and (
        parsed.scheme == "https"
        or (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"})
    )
    if not valid_endpoint:
        report.update({
            "state": "invalid",
            "error": "OPENAI_BROKER_ENDPOINT_INVALID",
            "action": "远程 provider 使用 HTTPS；本地服务只能使用 localhost/127.0.0.1/::1。",
        })
        return report
    credential = _credential_status(config)
    ready = bool(credential.get("configured"))
    report.update({
        "state": "ready" if ready else "blocked",
        "provider": parsed.hostname,
        "model": config["model"],
        "api_style": config["api_style"],
        "auth_required": bool(config.get("auth_required", True)),
        "credential": credential,
    })
    if ready:
        report["next_step"] = "可以运行 verdi run ... --llm-config PATH，或在交互会话中输入 /llm 查看帮助。"
    else:
        report["action"] = "补充 token 文件或环境变量后重新运行 verdi llm status。"
    return report


def render_llm_guide(*, config_path: Path | None = None) -> str:
    """Return the copyable first-contact guide with the active config path."""

    path = (config_path or default_config_path()).expanduser()
    return (
        GUIDE.replace("CONFIG_DIR", str(path.parent))
        .replace("CONFIG_PATH", str(path))
        .replace("AUTH_PATH", str(path.parent / "auth"))
    )
