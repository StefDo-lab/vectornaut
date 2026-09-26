# -*- coding: utf-8 -*-
"""
Step-by-step "human (or other model) in the loop" stand-in for the Gemini client.

The pipeline runs in live mode, but every model call is answered from a session
folder instead of the Gemini API:

    python -m vectornaut.llm_replay --session runs/riblet --query "..."

Each invocation replays the answers already present in <session>/responses/ and stops
at the first call without an answer. That call's full prompt and the JSON schema of
the expected response are written to <session>/requests/NN_<stage>.md. Write the answer
as JSON to <session>/responses/NN.json and run the same command again. When every call
is answered, the pipeline result is written to <session>/result.json and the Markdown
report to <session>/report.md.

Runtime data (history, saved models, generated scripts) goes to <session>/data, so
reruns reuse the same cached models and stay reproducible. Deterministic analytical (SymPy)
solves are memoised in <session>/solver_memo (keyed by equation, boundary conditions, parameter
values, domain and solver code; see vectornaut.solvers.solvers_1d.solve_analytical), so a rerun
does not repeat expensive symbolic solves or symbolic timeouts; --no-solver-memo turns it off.

The idea-space explorer (vectornaut/explorer) can be driven the same way:

    python -m vectornaut.llm_replay --explorer --session runs/x --profile business \
        --query "..." --rounds 2 --batch 4 [--seed 0]

Here --rounds counts explorer rounds (pipeline optimization rounds per materials
candidate: --opt-rounds). Every invocation rebuilds the explorer archive from scratch in
<session>/data/explorer, so the search orders, and therefore the prompts, are the same
on each rerun. The result summary goes to <session>/result.json, the cumulative map
report to <session>/report.md and all reports to <session>/explorer_reports/.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from contextlib import ExitStack
from typing import Any, Dict, Optional
from unittest import mock

# Modules that obtain a client through get_client().
CLIENT_MODULES = (
    "vectornaut.miner",
    "vectornaut.formulator",
    "vectornaut.auditor",
    "vectornaut.optimizer",
    "vectornaut.synthesizer",
    "vectornaut.script_generator",
    "vectornaut.test_generator",
)
# Explorer modules that obtain a client through get_client() (generator, business and materials critic).
EXPLORER_CLIENT_MODULES = (
    "vectornaut.explorer.generator",
    "vectornaut.explorer.profiles.business",
    "vectornaut.explorer.profiles.materials",
)


class NeedResponse(BaseException):
    # BaseException so the pipeline's broad `except Exception` fallbacks don't swallow it.
    def __init__(self, index: int, request_path: str):
        super().__init__(f"Call {index:02d} needs a response (see {request_path})")
        self.index = index
        self.request_path = request_path


class _Response:
    def __init__(self, parsed: Any):
        self.parsed = parsed
        self.text = parsed.model_dump_json() if hasattr(parsed, "model_dump_json") else json.dumps(parsed)


def _schema_of(config: Any) -> Any:
    return getattr(config, "response_schema", None) if config is not None else None


def _thinking_of(config: Any) -> Optional[str]:
    thinking = getattr(config, "thinking_config", None) if config is not None else None
    level = getattr(thinking, "thinking_level", None) if thinking is not None else None
    return getattr(level, "value", level)


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


class _Models:
    def __init__(self, client: "ReplayClient"):
        self._client = client

    def generate_content(self, model: str = None, contents: Any = None, config: Any = None, **kwargs):
        return self._client.answer(model, contents, config)


class ReplayClient:
    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.requests_dir = os.path.join(session_dir, "requests")
        self.responses_dir = os.path.join(session_dir, "responses")
        os.makedirs(self.requests_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
        self.calls = 0
        self.log = []
        self.models = _Models(self)

    def answer(self, model: str, contents: Any, config: Any):
        self.calls += 1
        index = self.calls
        schema = _schema_of(config)
        stage = schema.__name__ if isinstance(schema, type) else "text"
        prompt = contents if isinstance(contents, str) else json.dumps(contents, ensure_ascii=False, default=str)
        prompt_hash = _prompt_hash(prompt)

        response_path = os.path.join(self.responses_dir, f"{index:02d}.json")
        request_path = os.path.join(self.requests_dir, f"{index:02d}_{stage}.md")
        entry = {"index": index, "stage": stage, "model": model, "thinking": _thinking_of(config), "prompt_hash": prompt_hash}
        self.log.append(entry)

        if os.path.exists(response_path):
            with open(response_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            answered_hash = data.pop("_prompt_hash", None) if isinstance(data, dict) else None
            if answered_hash and answered_hash != prompt_hash:
                print(f"[replay] Warning: prompt of call {index:02d} changed since it was answered.")
            parsed = schema.model_validate(data) if hasattr(schema, "model_validate") else data
            return _Response(parsed)

        schema_json = schema.model_json_schema() if hasattr(schema, "model_json_schema") else None
        with open(request_path, "w", encoding="utf-8") as f:
            f.write(f"# Call {index:02d}: {stage}\n\n")
            f.write(f"- Model requested: `{model}`\n- Thinking level: `{entry['thinking']}`\n")
            f.write(f"- Prompt hash: `{prompt_hash}` (optionally store it as `_prompt_hash` in the response)\n")
            f.write(f"- Answer file: `responses/{index:02d}.json`\n\n## Prompt\n\n")
            f.write(prompt.strip() + "\n")
            if schema_json is not None:
                f.write("\n## Response JSON schema\n\n```json\n")
                f.write(json.dumps(schema_json, indent=2, ensure_ascii=False))
                f.write("\n```\n")
        raise NeedResponse(index, request_path)


SOLVER_MEMO_ENV = "VECTORNAUT_SOLVER_MEMO_DIR"


def solver_memo_env(session_dir: str, enabled: bool = True) -> Dict[str, str]:
    """Environment for the solver memo of a replay session (<session>/solver_memo), or {} when disabled."""
    return {SOLVER_MEMO_ENV: os.path.join(session_dir, "solver_memo")} if enabled else {}


def run_session(session_dir: str, query: str, epochs: int, rounds: int, solver_memo: bool = True) -> Dict[str, Any]:
    session_dir = os.path.abspath(session_dir)
    os.makedirs(session_dir, exist_ok=True)
    client = ReplayClient(session_dir)
    env = {
        "VECTORNAUT_DATA_DIR": os.path.join(session_dir, "data"),
        # The pipeline only runs live when a key is present; the replay client never uses it.
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY") or "replay-session",
        **solver_memo_env(session_dir, solver_memo),
    }

    from vectornaut.pipeline import PipelineRunRequest, run_pipeline

    with ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, env))
        for module in CLIENT_MODULES:
            stack.enter_context(mock.patch(f"{module}.get_client", return_value=client))
        try:
            result = run_pipeline(PipelineRunRequest(
                query=query,
                epochs=epochs,
                is_mock=False,
                max_optimization_rounds=rounds,
            ))
            # Archive like POST /api/run does; this also renders result["report_md"].
            from vectornaut.reporting import archive_run_data
            archive_run_data(result)
            status = {"status": "complete", "calls": client.calls}
            if result.get("status") == "rejected":
                # Physically infeasible request: a structured result, not a failure.
                status = {"status": "rejected", "calls": client.calls, "rejection": result.get("rejection")}
        except NeedResponse as need:
            result = None
            status = {"status": "needs_response", "index": need.index, "request": need.request_path, "calls": client.calls}
        except Exception as exc:
            result = None
            status = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "calls": client.calls}

    with open(os.path.join(session_dir, "calls.json"), "w", encoding="utf-8") as f:
        json.dump(client.log, f, indent=2, ensure_ascii=False)
    if result is not None:
        with open(os.path.join(session_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        with open(os.path.join(session_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(result.get("report_md") or "")
    return status


def run_explorer_session(
    session_dir: str,
    profile: str,
    query: str,
    rounds: int,
    batch: int,
    seed: int = 0,
    weights: Optional[str] = None,
    epochs: int = 40,
    opt_rounds: int = 1,
    use_critic: bool = True,
    solver_memo: bool = True,
) -> Dict[str, Any]:
    """
    Explorer run whose model calls (generator, critic, pipeline stages) are answered from the session.
    ``solver_memo``: memoise analytical solves in <session>/solver_memo (kept across reruns).
    """
    session_dir = os.path.abspath(session_dir)
    os.makedirs(session_dir, exist_ok=True)
    client = ReplayClient(session_dir)
    data_dir = os.path.join(session_dir, "data")
    # Deterministic replays: the archive is rebuilt from the recorded answers on every invocation.
    shutil.rmtree(os.path.join(data_dir, "explorer"), ignore_errors=True)
    shutil.rmtree(os.path.join(session_dir, "explorer_reports"), ignore_errors=True)
    env = {
        "VECTORNAUT_DATA_DIR": data_dir,
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY") or "replay-session",
        **solver_memo_env(session_dir, solver_memo),
    }

    from vectornaut.explorer.profiles import get_profile
    from vectornaut.explorer.run import ExplorerRunner
    from vectornaut.explorer.strategies import parse_weights

    summary = None
    with ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, env))
        for module in CLIENT_MODULES + EXPLORER_CLIENT_MODULES:
            stack.enter_context(mock.patch(f"{module}.get_client", return_value=client))
        try:
            runner = ExplorerRunner(
                get_profile(profile), query, seed=seed, weights=parse_weights(weights), mock=False,
                client=client, out_dir=os.path.join(session_dir, "explorer_reports"), epochs=epochs,
                opt_rounds=opt_rounds, use_critic=use_critic,
            )
            summary = runner.run(rounds, batch)
            status = {"status": "complete", "calls": client.calls, "elites": summary["elites"]}
        except NeedResponse as need:
            status = {"status": "needs_response", "index": need.index, "request": need.request_path, "calls": client.calls}
        except Exception as exc:
            status = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "calls": client.calls}

    with open(os.path.join(session_dir, "calls.json"), "w", encoding="utf-8") as f:
        json.dump(client.log, f, indent=2, ensure_ascii=False)
    if summary is not None:
        with open(os.path.join(session_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        with open(summary["reports"]["map"], "r", encoding="utf-8") as src:
            report_md = src.read()
        with open(os.path.join(session_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(report_md)
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="Session folder holding requests/ and responses/")
    parser.add_argument("--query", required=True, help="Design prompt for the pipeline")
    parser.add_argument("--epochs", type=int, default=None, help="PINN epochs (default 200; explorer: 40)")
    parser.add_argument("--rounds", type=int, default=2,
                        help="Maximum optimization rounds (with --explorer: explorer rounds)")
    explorer = parser.add_argument_group("explorer (with --explorer)")
    explorer.add_argument("--explorer", action="store_true", help="Drive the idea-space explorer instead of one pipeline run")
    explorer.add_argument("--profile", choices=("materials", "business"), default=None)
    explorer.add_argument("--batch", type=int, default=4, help="Candidates per explorer round")
    explorer.add_argument("--seed", type=int, default=0)
    explorer.add_argument("--strategy-weights", default=None)
    explorer.add_argument("--opt-rounds", type=int, default=1, help="Pipeline optimization rounds per materials candidate")
    explorer.add_argument("--no-critic", action="store_true", help="Business profile: skip the critic call")
    parser.add_argument("--no-solver-memo", action="store_true",
                        help="Do not memoise analytical solves in <session>/solver_memo")
    args = parser.parse_args(argv)

    if args.explorer:
        if not args.profile:
            parser.error("--explorer needs --profile")
        status = run_explorer_session(
            args.session, args.profile, args.query, args.rounds, args.batch, seed=args.seed,
            weights=args.strategy_weights, epochs=args.epochs or 40, opt_rounds=args.opt_rounds,
            use_critic=not args.no_critic, solver_memo=not args.no_solver_memo,
        )
    else:
        status = run_session(args.session, args.query, args.epochs or 200, args.rounds,
                             solver_memo=not args.no_solver_memo)
    print(json.dumps(status, ensure_ascii=False))
    return 0 if status["status"] in ("complete", "rejected", "needs_response") else 1


if __name__ == "__main__":
    sys.exit(main())
