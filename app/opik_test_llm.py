#!/usr/bin/env python3
"""
Minimal script to verify Opik tracing: one LLM call, wrapped with Opik, then flush.

Run from the app directory with your venv activated:
  cd app && source venv/bin/activate && python opik_test_llm.py

Requires: .env with OPENAI_API_KEY, OPIK_API_KEY, OPIK_WORKSPACE.
Opik targets Cloud automatically when api_key and workspace are set (do not set OPIK_URL_OVERRIDE).
Check your Opik/Comet dashboard after running to see the trace.
"""

import os
import sys
from pathlib import Path

# Ensure we're in app directory and load .env first (before any opik/openai imports)
APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)
sys.path.insert(0, str(APP_DIR))

try:
    from dotenv import load_dotenv
    # override=True so .env wins over any OPENAI_API_KEY already set in the shell (avoids 401 from stale key)
    load_dotenv(APP_DIR / ".env", override=True)
except ImportError:
    pass  # .env not loaded; use exported env vars or opik configure

# Configure Opik before creating/wrapping the OpenAI client
try:
    import opik
    api_key = os.environ.get("OPIK_API_KEY")
    workspace = os.environ.get("OPIK_WORKSPACE")
    if api_key and workspace:
        opik.configure(
            api_key=api_key,
            workspace=workspace,
            use_local=False,
        )
        print("Opik configured (API key + workspace from .env)")
    else:
        opik.configure()
        print("Opik configured (using existing config e.g. ~/.opik.config)")
except Exception as e:
    print("Opik configure failed:", e)
    sys.exit(1)

# Create OpenAI client and wrap with Opik
from openai import OpenAI
from utils.opik_helper import wrap_openai_for_opik, flush_opik

# OpenAI key from .env (strip whitespace/quotes; 401 = key invalid/expired — check platform.openai.com)
openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
if not openai_key:
    print("OPENAI_API_KEY not set in .env. Add a valid key from https://platform.openai.com/account/api-keys")
    sys.exit(1)
client = OpenAI(api_key=openai_key)
client = wrap_openai_for_opik(client, project_name=os.environ.get("OPIK_PROJECT_NAME", "METAI"))

# Single LLM call
print("Sending one LLM request...")
resp = client.chat.completions.create(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    messages=[{"role": "user", "content": "What is 2+2? Reply in one short sentence."}],
    max_tokens=100,
)
answer = resp.choices[0].message.content if resp.choices else "(no response)"
print("Answer:", answer)

# Flush so the trace is sent to Opik dashboard
flush_opik()
print("Opik flush done. Check your Opik/Comet dashboard for the trace.")
