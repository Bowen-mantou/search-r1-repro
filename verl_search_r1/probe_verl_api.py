"""Pre-flight probe for E6-B (run BEFORE spending GPU hours).

Checks, in order, everything that could fail at runtime:
  1. veRL agent-loop API — base signature, registry, output fields
     (FAIL ① fix: ``_agent_loop_registry`` replaces removed ``get_agent_loop_class``)
  2. SearchR1AgentLoop instantiation the way AgentLoopWorker does it —
     DictConfigWrap + real tokenizer + chat-template smoke test
  3. Hydra config compose (FAIL ② fix: config dir = dirname(main_ppo.py)/config)
     + key-drift scan of every override against the BASE config
     + full compose with overrides + wiring asserts
  4. Reward function — import, signature, unit smoke on a fake DataProto
  5. Library versions + GPU + flashinfer
  6. Data files + columns + e6b_run.sh sanity (no stale --config-path etc.)
  7. (--live-search) one real DeepSeek query with full error details (FAIL ③)

Usage (on server, from $VERL_DIR):
    PROJECT_DIR=~/autodl-tmp/search-r1 python $PROJECT_DIR/verl_search_r1/probe_verl_api.py --live-search
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", os.path.expanduser("~/autodl-tmp/search-r1")))
VERL_DIR = Path.cwd()

sys.path.insert(0, str(PROJECT_DIR))

OK, FAIL, WARN = "[OK] ", "[FAIL]", "[WARN]"
results: list[str] = []
_SENTINEL = object()
BASE_MODEL = os.environ.get("BASE_MODEL", "wang072266/qwen3.5-4b-search-r1-sft")

args: argparse.Namespace


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(f"  {OK if ok else FAIL} {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        print(f"  {FAIL} {name}  -- {detail}")


def warn(name: str, detail: str = "") -> None:
    results.append(f"  {WARN} {name}" + (f"  -- {detail}" if detail else ""))
    print(f"  {WARN} {name}" + (f"  -- {detail}" if detail else ""))


def find_config_dir() -> Path | None:
    """veRL main's hydra config dir: dirname(verl/trainer/main_ppo.py)/config."""
    try:
        import verl.trainer.main_ppo as _mppo  # noqa: F401

        d = Path(_mppo.__file__).resolve().parent / "config"
        if d.is_dir():
            return d
    except Exception:  # noqa: BLE001  (e.g. transfer_queue missing — fall through)
        pass
    for cand in (
        VERL_DIR / "verl" / "trainer" / "config",
        VERL_DIR / "verl" / "experimental" / "trainer" / "config",
    ):
        if cand.is_dir():
            return cand
    hits = sorted(VERL_DIR.rglob("trainer/config/ppo_trainer.yaml"))
    return hits[0].parent if hits else None


# ═══════════════════════════════════════════════════════════════════════════
# Sections
# ═══════════════════════════════════════════════════════════════════════════


def section_agent_loop_api() -> None:
    print("\n[1/7] veRL agent-loop API")
    try:
        from verl.experimental.agent_loop.agent_loop import (  # type: ignore
            AgentLoopBase,
            AgentLoopOutput,
            _agent_loop_registry,
        )
    except ImportError as e:
        check("veRL agent_loop import", False, repr(e))
        return

    import verl_search_r1.search_agent_loop  # noqa: F401  (registers the loop)

    check(
        "AgentLoopBase.__init__ signature",
        True,
        str(inspect.signature(AgentLoopBase.__init__)),
    )
    fields = getattr(AgentLoopOutput, "model_fields", None) or getattr(
        AgentLoopOutput, "__dataclass_fields__", {}
    )
    check("AgentLoopOutput is pydantic model", hasattr(AgentLoopOutput, "model_fields"))
    required = ("prompt_ids", "response_ids", "response_mask", "metrics", "extra_fields")
    check(
        "AgentLoopOutput fields",
        all(f in fields for f in required),
        f"fields={sorted(fields)}",
    )
    entry = _agent_loop_registry.get("search_r1_agent")
    check(
        "registry has 'search_r1_agent'",
        entry is not None,
        f"registry keys={sorted(_agent_loop_registry)}",
    )


def section_instantiation() -> tuple[Any, Any]:
    print("\n[2/7] SearchR1AgentLoop instantiation (worker-style)")
    loop = None
    agent_kwargs: dict[str, Any] = {}
    try:
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer

        # veRL main 0.10: DictConfigWrap lives in the agent_loop module (the
        # base-class signature references it), not verl.utils.config.
        try:
            from verl.experimental.agent_loop.agent_loop import DictConfigWrap
        except ImportError:  # older locations
            from verl.utils.config import DictConfigWrap

        agent_yaml = PROJECT_DIR / "verl_search_r1" / "agent_config.yaml"
        if not agent_yaml.exists():
            check("agent_config.yaml exists", False, str(agent_yaml))
            return None, None
        entries = OmegaConf.load(agent_yaml)
        for entry in entries:
            if entry.get("name") == "search_r1_agent":
                agent_kwargs = {k: v for k, v in entry.items() if k not in ("name", "_target_")}
                break
        check(
            "agent_config.yaml list has search_r1_agent entry",
            bool(agent_kwargs),
            f"entries={[e.get('name') for e in entries]}",
        )
        if not agent_kwargs:
            return None, None

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        check("tokenizer loaded", True, BASE_MODEL)

        trainer_config = DictConfigWrap(
            OmegaConf.create(
                {
                    "actor_rollout_ref": {
                        "rollout": {
                            "agent": {"default_agent_loop": "search_r1_agent"},
                            "response_length": 4096,
                            "max_model_len": 8192,
                            "multi_turn": {"enable": True},
                        }
                    }
                }
            )
        )
        data_config = DictConfigWrap(OmegaConf.create({}))

        import verl_search_r1.search_agent_loop as sal

        loop = sal.SearchR1AgentLoop(
            trainer_config=trainer_config,
            server_manager=None,
            tokenizer=tokenizer,
            processor=None,
            dataset_cls=None,
            data_config=data_config,
            hf_model_type=None,
            **agent_kwargs,
        )
        check("SearchR1AgentLoop instantiation (worker-style)", True)
        check("loop.loop present", hasattr(loop, "loop"))
        check("rollout_config resolved", getattr(loop, "rollout_config", None) is not None)
        ids = loop._apply_chat_template([{"role": "user", "content": "今天的天气怎么样？"}])
        check(
            "chat-template smoke (tools + enable_thinking)",
            isinstance(ids, list) and len(ids) > 50,
            f"type={type(ids).__name__} len={len(ids)} head={ids[:30]}",
        )
    except Exception as e:  # noqa: BLE001
        check("SearchR1AgentLoop instantiation", False, repr(e))
    return loop, agent_kwargs


def section_config() -> None:
    print("\n[3/7] Hydra config compose + override key-drift scan")
    overrides_file = Path(args.overrides_file)
    if not overrides_file.exists():
        check("overrides file exists", False, f"{overrides_file} — run e6b_run.sh --probe")
        return
    overrides = [
        ln.strip()
        for ln in overrides_file.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    check("overrides loaded", True, f"{len(overrides)} from {overrides_file.name}")
    if not overrides:
        return

    config_dir = find_config_dir()
    check("hydra config dir exists", config_dir is not None, str(config_dir))
    if config_dir is None:
        return

    try:
        from hydra import compose, initialize_config_dir

        import omegaconf

        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            compose(config_name="ppo_trainer")
        check("base compose (ppo_trainer) succeeded", True)

        # Key-drift gate: hydra hard-fails on any non-`+` override whose key is
        # missing from the base config, so composing WITH the full override list
        # below IS the drift check. `+`-key placement is covered by the wiring
        # asserts (reward_kwargs.enable_prm_lite). No separate base scan:
        # OmegaConf `???` placeholders (e.g. rollout.name) produce false positives.
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            full = compose(config_name="ppo_trainer", overrides=overrides)
        check("compose with full overrides (key-drift gate)", True)

        r = full.actor_rollout_ref
        check("rollout.name=vllm", r.rollout.name == "vllm", str(r.rollout.name))
        check(
            "rollout.max_model_len=8192",
            r.rollout.max_model_len == 8192,
            str(r.rollout.max_model_len),
        )
        check("default_agent_loop", r.rollout.agent.default_agent_loop == "search_r1_agent")
        check("multi_turn.enable", bool(r.rollout.multi_turn.enable))
        check("reward.reward_model.enable is False", full.reward.reward_model.enable is False)
        cfn = full.reward.custom_reward_function
        check("custom reward name", cfn.name == "search_r1_reward", str(cfn.name))
        check("custom reward path", cfn.path == "verl_search_r1.reward_fn", str(cfn.path))
        prm = cfn.reward_kwargs.get("enable_prm_lite", _SENTINEL)
        check("reward_kwargs.enable_prm_lite wired", isinstance(prm, bool), f"prm={prm}")
        check("data.truncation=left", full.data.truncation == "left", str(full.data.truncation))
        check("data.shuffle=True", bool(full.data.shuffle))
        check("ref.log_prob_use_dynamic_bsz", bool(r.ref.log_prob_use_dynamic_bsz))
        check("adv_estimator=grpo", full.algorithm.adv_estimator == "grpo")
        check("n_gpus_per_node", int(full.trainer.n_gpus_per_node), str(full.trainer.n_gpus_per_node))

        with open(VERL_DIR / ".e6b_composed.yaml", "w", encoding="utf-8") as fh:
            omegaconf.OmegaConf.save(full, fh)
        print(f"  composed config saved to {VERL_DIR / '.e6b_composed.yaml'}")
    except Exception as e:  # noqa: BLE001
        check("config compose", False, repr(e))


def section_reward() -> None:
    print("\n[4/7] Reward function")
    try:
        from verl_search_r1 import reward_fn

        sig = inspect.signature(reward_fn.search_r1_reward)
        has_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        check("search_r1_reward accepts **kwargs", has_kw, str(sig))

        import torch

        fake_batch = {
            "question": ["测试问题"],
            "reward_model": [{"ground_truth": ["答案"], "style": "rule"}],
            "data_source": ["probe"],
            "tool_extra_fields": [{"response_text": "答案是：答案", "search_calls": 1}],
        }
        fake_data = SimpleNamespace(
            batch=torch.zeros(1, 4, dtype=torch.long),
            non_tensor_batch=fake_batch,
        )
        out = reward_fn.search_r1_reward(fake_data)
        ok = (
            isinstance(out, dict)
            and "reward_tensor" in out
            and out["reward_tensor"].shape[0] == 1
        )
        check(
            "reward fn smoke run (non_tensor_batch + tool_extra_fields)",
            ok,
            str({k: out[k] for k in out if k != "reward_tensor"}),
        )
    except Exception as e:  # noqa: BLE001
        check("reward fn", False, repr(e))


def section_versions() -> None:
    print("\n[5/7] Library versions + GPU")
    import torch

    check("torch", True, f"{torch.__version__} cuda={torch.cuda.is_available()}")
    ngpu = torch.cuda.device_count()
    if ngpu >= 2:
        check("CUDA GPUs >= 2", True, f"{ngpu} GPUs")
    else:
        warn("CUDA GPUs < 2 — only N_GPUS=1 downgrade mode is viable", f"{ngpu} GPUs")
    try:
        import transformers

        check("transformers", True, transformers.__version__)
    except Exception as e:  # noqa: BLE001
        check("transformers", False, repr(e))
    try:
        import vllm

        check("vllm", True, f"vllm={vllm.__version__}")
    except Exception as e:  # noqa: BLE001
        check("vllm import", False, repr(e))
    try:
        import verl

        check("verl", True, f"verl={getattr(verl, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        check("verl import", False, repr(e))
    try:
        import transfer_queue  # noqa: F401

        # v1 trainer hard dep (git source: Ascend/TransferQueue, pyproject extra
        # "transferqueue"); missing → TaskRunnerV1 crashes at step 0.
        check("transfer_queue import (v1 trainer hard dep)", True)
    except Exception as e:  # noqa: BLE001
        check(
            "transfer_queue import",
            False,
            f"{repr(e)} — pip install 'TransferQueue @ git+https://github.com/Ascend/TransferQueue.git@main'",
        )
    try:
        import flashinfer

        check(
            "flashinfer import (VLLM_ATTENTION_BACKEND=FLASHINFER)",
            True,
            getattr(flashinfer, "__version__", "?"),
        )
    except Exception as e:  # noqa: BLE001
        check("flashinfer import", False, repr(e))
    if os.environ.get("VLLM_ATTENTION_BACKEND", "").upper() != "FLASHINFER":
        warn(
            "VLLM_ATTENTION_BACKEND not FLASHINFER in probe env",
            "e6b_run.sh exports it at launch; probe runs before that",
        )


def section_data() -> None:
    print("\n[6/7] Data files + e6b_run.sh sanity")
    for name in ("train.parquet", "test.parquet"):
        p = PROJECT_DIR / "datasets" / name
        check(
            f"datasets/{name}",
            p.exists(),
            f"{p.stat().st_size // 1024 // 1024}MB" if p.exists() else "missing",
        )
    try:
        import pyarrow.parquet as pq

        cols = set(pq.read_schema(PROJECT_DIR / "datasets" / "train.parquet").names)
        need = {"id", "question", "answers", "data_source", "prompt", "reward_model"}
        check("train.parquet columns", need <= cols, f"cols={sorted(cols)}")
    except Exception as e:  # noqa: BLE001
        check("train.parquet columns", False, repr(e))

    run_sh = PROJECT_DIR / "verl_search_r1" / "e6b_run.sh"
    if run_sh.exists():
        txt = run_sh.read_text()
        check("e6b_run.sh: no stale --config-path=... arg", "--config-path=" not in txt)
        check("e6b_run.sh: exports PYTHONPATH", "export PYTHONPATH" in txt)
        check("e6b_run.sh: FLASHINFER backend", "FLASHINFER" in txt)
        check("e6b_run.sh: reward.* key path", "reward.custom_reward_function.path" in txt)
        check("e6b_run.sh: max_model_len=8192", "max_model_len=8192" in txt)
    else:
        check("e6b_run.sh present", False, str(run_sh))


def section_live_search(agent_kwargs: dict[str, Any] | None) -> None:
    print("\n[7/7] Live DeepSeek search (--live-search)")
    if not args.live_search:
        print("  skipped (pass --live-search to fire a real query)")
        return
    try:
        from search import create_search_client

        agent_kwargs = agent_kwargs or {}
        env_path = PROJECT_DIR / ".env"
        check(".env present", env_path.exists(), f"search key lives in {env_path}")
        client = create_search_client(
            agent_kwargs.get("search_backend", "deepseek"),
            str(env_path) if env_path.exists() else None,
            model=agent_kwargs.get("search_model", "deepseek-v4-flash"),
            timeout=float(agent_kwargs.get("search_timeout", 60.0)),
        )
        t0 = time.time()
        r = client.search("Baybrook Mall")
        dt = time.time() - t0
        ok = bool(r.ok) and len(r.items) > 0
        check(
            "live search query",
            ok,
            f"{dt:.1f}s items={len(r.items)} latency={getattr(r, 'latency', None)} "
            f"status={getattr(r, 'status', None)} error={getattr(r, 'error', None)!r}",
        )
    except Exception as e:  # noqa: BLE001
        check("live search query", False, repr(e))


# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-search", action="store_true")
    parser.add_argument("--overrides-file", default=str(VERL_DIR / ".e6b_overrides.txt"))
    args = parser.parse_args()

    print("=" * 72)
    print(f"E6-B pre-flight probe  (PROJECT_DIR={PROJECT_DIR}, VERL_DIR={VERL_DIR})")
    print(f"BASE_MODEL={BASE_MODEL}")
    print("=" * 72)

    section_agent_loop_api()
    _loop, agent_kwargs = section_instantiation()
    section_config()
    section_reward()
    section_versions()
    section_data()
    section_live_search(agent_kwargs)

    print("\n" + "=" * 72)
    n_fail = sum(1 for r in results if FAIL in r)
    n_warn = sum(1 for r in results if WARN in r)
    print(f"Probe done: {len(results)} checks, {n_fail} failures, {n_warn} warnings")
    for r in results:
        print(r)
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
