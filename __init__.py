import os
import sys
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── auto-update from GitHub on every ComfyUI startup ─────────────────────────
# Runs git pull BEFORE importing nodes.py so any changes land in this session.
# Silent fallback on any failure — never blocks startup.
try:
    _repo = os.path.dirname(os.path.abspath(__file__))
    _r = subprocess.run(
        ["git", "-C", _repo, "pull", "--rebase", "--autostash"],
        capture_output=True, text=True, timeout=30
    )
    if _r.returncode == 0:
        if "Already up to date." not in _r.stdout:
            print(f"[Cosmos3] Auto-updated from GitHub:\n{_r.stdout.strip()}")
        # else: nothing to say
    else:
        print(f"[Cosmos3] git pull failed (non-fatal): {_r.stderr.strip()[:200]}")
except Exception as _e:
    print(f"[Cosmos3] git pull skipped: {_e}")
# ─────────────────────────────────────────────────────────────────────────────

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
