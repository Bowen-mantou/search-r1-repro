"""E6-B training entry point.

Imports SearchR1AgentLoop (which @registers "search_r1_agent" into veRL's
agent-loop registry) BEFORE starting the trainer, then delegates to veRL's
hydra entry point (``verl.trainer.main_ppo.main``).

This file is copied to $VERL_DIR by e6b_run.sh. No ``--config-path`` is passed:
hydra resolves the decorator's ``config_path="config"`` relative to
``verl/trainer/main_ppo.py``, i.e. ``$VERL_DIR/verl/trainer/config/``.
PYTHONPATH must include $PROJECT_DIR so Ray workers can import
``verl_search_r1`` (exported by e6b_run.sh).

Launch (from inside $VERL_DIR):
    PROJECT_DIR=~/autodl-tmp/search-r1 python e6b_main.py \
        --config-name=ppo_trainer ...overrides...
"""
import os
import sys

_PROJECT_DIR = os.environ.get("PROJECT_DIR", os.path.expanduser("~/autodl-tmp/search-r1"))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)
if "PYTHONPATH" not in os.environ or _PROJECT_DIR not in os.environ["PYTHONPATH"]:
    os.environ["PYTHONPATH"] = _PROJECT_DIR + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )

# Register "search_r1_agent" before verl's rollout looks it up.
import verl_search_r1.search_agent_loop  # noqa: F401

from verl.trainer.main_ppo import main  # noqa: E402

main()
