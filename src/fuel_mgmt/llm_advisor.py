"""调用 DeepSeek（OpenAI 兼容模式）生成管理建议。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import API_KEY_FILENAME, DEEPSEEK_BASE_URL, DEFAULT_DEEPSEEK_MODEL

SYSTEM_PROMPT = """你是营运车队的油料管理员。下面是一辆货车若干天油耗轨迹的结构化分析结果，不是原始采样点。
请只依据这些事实给出可执行的管理建议，不要编造数据中不存在的加油次数、地点或数字。
若某项证据不足，明确写“证据不足”。

务必区分：
- confidence=高可信 的加油才可作为确定加油；
- 存疑事件不得当成已核实加油；
- steep_drops 才是短时流失候选，长停车净变化不是盗油证据。

请用简体中文，按以下五个小节输出（使用 Markdown 二级标题）：

## 总体判断
## 需核查事项
## 加油策略
## 驾驶与调度观察
## 传感器维护

每节写 3–8 句，具体、可执行，避免空话。"""


def _project_root() -> Path:
    """作业根目录。"""
    return Path(__file__).resolve().parents[2]


def load_api_key() -> str:
    """读取 DeepSeek API Key。

    优先环境变量 `DEEPSEEK_API_KEY`，否则读取作业根目录的 `deepseek-api.txt`。
    """
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    path = _project_root() / API_KEY_FILENAME
    if path.exists():
        return path.read_text(encoding="utf-8-sig").strip()
    cwd_path = Path.cwd() / API_KEY_FILENAME
    if cwd_path.exists():
        return cwd_path.read_text(encoding="utf-8-sig").strip()
    return ""


def generate_advice(summary: dict[str, Any]) -> str | None:
    """根据分析摘要调用 DeepSeek；无 Key 或失败时返回 None。

    只发送 summary JSON，不发送原始 1 万余行轨迹。

    Args:
        summary: `build_summary` 的输出。

    Returns:
        Markdown 文本，或 None。
    """
    api_key = load_api_key()
    if not api_key:
        return None

    model = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL).strip() or DEFAULT_DEEPSEEK_MODEL
    payload = json.dumps(summary, ensure_ascii=False)
    user_content = "分析摘要 JSON 如下：\n\n```json\n" + payload + "\n```"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
        )
        text = resp.choices[0].message.content
        if not text or not str(text).strip():
            return None
        return str(text).strip()
    except Exception as exc:  # noqa: BLE001 — 建议章失败不得中断报告
        return f"__LLM_ERROR__:{exc.__class__.__name__}: {exc}"
