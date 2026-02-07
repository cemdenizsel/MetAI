"""
Opik (Comet) integration for LLM observability and tracing.

Matches the pattern from https://www.comet.com/docs/opik/tracing/log_traces:
- configure() so the SDK has API key and workspace (from env or CLI config)
- track_openai(client) to wrap the OpenAI client
- flush after requests so traces appear in the dashboard

Set OPIK_API_KEY and OPIK_WORKSPACE in .env (or run `opik configure`).
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

OPIK_AVAILABLE = False
track_openai_fn = None
_configured = False

try:
    from opik.integrations.openai import track_openai as _track_openai
    track_openai_fn = _track_openai
    OPIK_AVAILABLE = True
except ImportError:
    logger.debug("Opik not installed; LLM tracing disabled. Install with: pip install opik")
except Exception as e:
    logger.debug("Opik import failed: %s", e)


def _ensure_configured() -> None:
    """Call opik.configure() once if not already done (e.g. when main.py didn't run)."""
    global _configured
    if not OPIK_AVAILABLE or _configured:
        return
    try:
        import opik
        if hasattr(opik, "configure") and callable(opik.configure):
            api_key = os.environ.get("OPIK_API_KEY")
            workspace = os.environ.get("OPIK_WORKSPACE")
            if api_key and workspace:
                opik.configure(
                    api_key=api_key,
                    workspace=workspace,
                    use_local=False,
                )
                _configured = True
                logger.info("Opik configured for Cloud (api_key, workspace)")
            else:
                opik.configure()
                _configured = True
    except Exception as e:
        logger.debug("Opik configure skipped: %s", e)


def wrap_openai_for_opik(client: Any, project_name: str = "metai") -> Any:
    """
    Wrap an OpenAI client with Opik tracing (same pattern as Opik docs).

    Requires opik package and OPIK_API_KEY + OPIK_WORKSPACE in env (or `opik configure`).
    Calls configure() once so the SDK is ready; then wraps with track_openai(client).
    """
    if client is None:
        return None
    if not OPIK_AVAILABLE or track_openai_fn is None:
        return client
    try:
        _ensure_configured()
        if not os.environ.get("OPIK_PROJECT_NAME"):
            os.environ["OPIK_PROJECT_NAME"] = project_name
        wrapped = track_openai_fn(client)
        logger.info("Opik tracing enabled for OpenAI client (project: %s)", project_name)
        return wrapped
    except Exception as e:
        logger.warning("Opik wrap failed, using unwrapped client: %s", e)
        return client


def flush_opik() -> None:
    """
    Flush buffered Opik traces so they are sent to the dashboard.
    Call after an LLM request (e.g. after emotion analysis) so traces show up promptly.
    See: https://www.comet.com/docs/opik/tracing/log_traces (Flushing traces and spans).
    """
    if not OPIK_AVAILABLE:
        return
    try:
        import opik
        if hasattr(opik, "flush") and callable(opik.flush):
            opik.flush()
            logger.debug("Opik flush completed")
        elif hasattr(opik, "Opik"):
            client = opik.Opik()
            if hasattr(client, "flush") and callable(client.flush):
                client.flush()
                logger.debug("Opik client flush completed")
    except Exception as e:
        logger.debug("Opik flush skipped or failed: %s", e)
