"""Entrypoint: read the question, run the graph, return the brief.

Input (from Studio or `trase-os-sdk run-workflow --query "..."`):
    {"user_message": "I take lisinopril and Advil. Anything I should know?"}
or, to skip the model's extraction step:
    {"user_message": "...", "drugs": ["lisinopril", "Advil"]}
"""

from __future__ import annotations

import logging
from typing import Any

# Exactly one graph exported from this module, for build-time topology inspection.
from agent.graph import safety_graph  # noqa: F401

log = logging.getLogger(__name__)

DEFAULT_QUESTION = "I take lisinopril for blood pressure and Advil for back pain. Anything I should know?"


def _read_input() -> dict[str, Any]:
    try:
        from trase_os_sdk.sandbox import NoInputError, NotInASandboxError, read_input
    except ImportError:
        return {}
    try:
        value = read_input()
    except (NoInputError, NotInASandboxError):
        return {}
    if isinstance(value, str):
        return {"user_message": value}
    return value if isinstance(value, dict) else {}


def run() -> str:
    """Called by the platform with no arguments. The return value is the step output."""
    payload = _read_input()
    question = (payload.get("user_message") or payload.get("query") or "").strip()
    if not question:
        log.info("no question in the step input: using the built-in example")
        question = DEFAULT_QUESTION
    state: dict[str, Any] = {"question": question}
    if isinstance(payload.get("drugs"), list) and payload["drugs"]:
        state["drug_names"] = [str(d) for d in payload["drugs"]]
    return safety_graph.invoke(state)["brief"]
