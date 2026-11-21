"""Original side of Harbor parity for FrontierFinance: claude-code outside Harbor.

  run      Run claude-code once per query in a fresh container, on local Docker,
           a Modal sandbox or a Daytona sandbox (--backend). The image is the task's own
           environment/Dockerfile plus the install step Harbor's claude-code
           agent runs; the instruction text, model, claude-code version, flags,
           environment, resources and time limit are Harbor's too. Rerunning the
           same command skips queries that already have a final status and
           retries the rest.
  collect  Apply one rerun rule to either side and write system_summaries.json
           for the grader. Takes an original-side run directory or Harbor job
           directories (a job, then rerun jobs: later trials replace earlier ones).
  score    Merge the grader's per_item.json files (later files replace earlier
           queries) into the macro qualification rate, and list queries to regrade.

Usage (from the repository root). ANTHROPIC_API_KEY, or ANTHROPIC_AUTH_TOKEN for a
bearer-auth gateway, and an optional ANTHROPIC_BASE_URL come from the environment;
--backend modal also needs the modal package and a Modal login; --backend daytona
needs the daytona package (<1, as Harbor 0.27.0 pins it) and DAYTONA_API_KEY:

  python harbor_parity/run_claude_code.py run -d <harbor>/datasets/frontier-finance \\
      -o runs/original1 -m claude-haiku-5-5 --claude-code-version 2.1.330 -n 8
  python harbor_parity/run_claude_code.py collect runs/original1 \\
      --summaries runs/original1/system_summaries.json

Then grade with this repository's grader and harbor_parity/eval.parity.yaml, and
  python harbor_parity/run_claude_code.py score grader_results/per_item.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import shutil
import subprocess
import time
import tomllib
import uuid
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# The one rerun rule for both sides, read from claude-code's own stream log:
# these statuses are results and are scored (0 without a report). Anything else
# (an API error, error_during_execution, no result event) is an infrastructure
# failure and is rerun.
FINAL_STATUSES = frozenset({"success", "error_max_turns", "timeout"})
# Set to the model when a base URL is set (harbor 0.27.0, claude_code.py _model_alias_env).
ALIAS_VARS = (
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
)
# Host variables Harbor's claude-code agent reads into its flags or environment.
# Both sides run with them unset.
HOST_OVERRIDES = (
    "CLAUDE_CODE_MAX_TURNS",
    "CLAUDE_CODE_EFFORT_LEVEL",
    "MAX_THINKING_TOKENS",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING",
    *ALIAS_VARS,
)
# Harbor's claude-code install step (ensure_system_dependencies, then bootstrap.sh),
# appended to the task's environment/Dockerfile.
INSTALL_LAYER = """
RUN apt-get update && apt-get install -y --no-install-recommends curl bash nodejs npm procps \\
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh | bash -s -- {version} \\
    && /root/.local/bin/claude --version
ENV PATH="/root/.local/bin:$PATH"
WORKDIR /app
"""
FORWARDED_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")
# Harbor's claude-code setup and run commands; the instruction arrives on stdin.
AGENT_COMMAND = (
    "mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app $CLAUDE_CONFIG_DIR/shell-snapshots "
    "$CLAUDE_CONFIG_DIR/statsig $CLAUDE_CONFIG_DIR/todos $CLAUDE_CONFIG_DIR/skills && "
    "claude --verbose --output-format=stream-json --permission-mode=bypassPermissions --print "
    "2>&1 | tee /logs/agent/claude-code.txt"
)


@dataclass(frozen=True)
class Task:
    instruction: str
    timeout: float
    cpus: float
    memory_mb: int
    storage_mb: int


def dataset_tasks(dataset: Path) -> dict[str, Task]:
    """query_id -> instruction and agent limits from Harbor-generated tasks."""
    tasks = {}
    for task in sorted(
        p for p in dataset.iterdir() if (p / "instruction.md").is_file()
    ):
        config = tomllib.loads((task / "task.toml").read_text())
        tasks[config["metadata"]["source_query_id"]] = Task(
            (task / "instruction.md").read_text(),
            float(config["agent"]["timeout_sec"]),
            float(config["environment"]["cpus"]),
            int(config["environment"]["memory_mb"]),
            int(config["environment"]["storage_mb"]),
        )
    return tasks


def agent_env(model: str) -> dict[str, str]:
    """The variables Harbor's claude-code agent sets (harbor 0.27.0, claude_code.py run()), plus the keys."""
    env = {
        "ANTHROPIC_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "IS_SANDBOX": "1",
        "CLAUDE_CONFIG_DIR": "/logs/agent/sessions",
    }
    if os.environ.get("ANTHROPIC_BASE_URL"):
        env.update(dict.fromkeys(ALIAS_VARS, model))
    env.update({key: os.environ[key] for key in FORWARDED_KEYS if os.environ.get(key)})
    return env


def parse_status(log: Path) -> str:
    """Status from the final stream-json result event; "no_result" if there is none."""
    if not log.is_file():
        return "no_result"
    for line in reversed(log.read_text(errors="replace").splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            if event.get("is_error") and event.get("subtype") == "success":
                return "api_error"
            return event.get("subtype", "no_result")
    return "no_result"


# A backend runs AGENT_COMMAND for one task with `env`, leaves /app/report.md in
# query_dir/app and /logs/agent in query_dir/logs/agent, and returns
# (exit code, timed out).
Backend = Callable[[Path, Task, dict[str, str]], tuple[int, bool]]


def agent_image_dir(dataset: Path, version: str, out: Path) -> Path:
    """Build context: the task environment (identical across tasks) plus INSTALL_LAYER."""
    envs = {
        (p / "environment" / "Dockerfile").read_text()
        for p in dataset.iterdir()
        if (p / "environment" / "Dockerfile").is_file()
    }
    if len(envs) != 1:
        raise SystemExit(f"expected one task Dockerfile, found {len(envs)}")
    context = out / "agent-image"
    context.mkdir(parents=True, exist_ok=True)
    (context / "Dockerfile").write_text(
        envs.pop() + INSTALL_LAYER.format(version=version)
    )
    return context


def docker_backend(context: Path, version: str) -> Backend:
    image = f"ff-parity-claude-code:{version}"
    subprocess.run(["docker", "build", "-q", "-t", image, str(context)], check=True)

    def run(query_dir: Path, task: Task, env: dict[str, str]) -> tuple[int, bool]:
        name = f"ff-parity-{uuid.uuid4().hex[:12]}"
        cmd = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--name",
            name,
            f"--cpus={task.cpus}",
            f"--memory={task.memory_mb}m",
        ]
        cmd += [
            "-v",
            f"{(query_dir / 'app').resolve()}:/app",
            "-v",
            f"{(query_dir / 'logs').resolve()}:/logs",
        ]
        for key in env:
            cmd += ["-e", key]  # by name, so values (keys) stay off the command line
        cmd += [image, "bash", "-c", AGENT_COMMAND]
        try:
            with open(query_dir / "instruction.md") as stdin:
                proc = subprocess.run(
                    cmd,
                    stdin=stdin,
                    stdout=subprocess.DEVNULL,
                    env={**os.environ, **env},
                    timeout=task.timeout,
                    check=False,
                )
            return proc.returncode, False
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True, check=False)
            return -1, True

    return run


def modal_backend(context: Path, version: str) -> Backend:
    import modal

    image = modal.Image.from_dockerfile(context / "Dockerfile", context_dir=context)
    app = modal.App.lookup("ff-harbor-parity", create_if_missing=True)

    def fetch(sandbox, command: str) -> bytes:
        proc = sandbox.exec("bash", "-c", f"{command} | base64 -w0")
        data = proc.stdout.read()
        proc.wait()
        return base64.b64decode(data) if data.strip() else b""

    def run(query_dir: Path, task: Task, env: dict[str, str]) -> tuple[int, bool]:
        sandbox = modal.Sandbox.create(
            "sleep",
            "infinity",
            app=app,
            image=image,
            cpu=task.cpus,
            memory=task.memory_mb,
            timeout=int(task.timeout) + 900,
        )
        try:
            start = time.monotonic()
            proc = sandbox.exec(
                "bash",
                "-c",
                AGENT_COMMAND,
                secrets=[modal.Secret.from_dict(env)],
                timeout=int(task.timeout),
            )
            proc.stdin.write(task.instruction)
            proc.stdin.write_eof()
            proc.stdin.drain()
            proc.stdout.read()
            code = proc.wait()
            timed_out = time.monotonic() - start >= task.timeout - 5
            agent_dir = query_dir / "logs" / "agent"
            (query_dir / "app" / "report.md").write_bytes(
                fetch(sandbox, "cat /app/report.md 2>/dev/null")
            )
            (agent_dir / "claude-code.txt").write_bytes(
                fetch(sandbox, "cat /logs/agent/claude-code.txt")
            )
            (agent_dir / "sessions.tar.gz").write_bytes(
                fetch(sandbox, "tar -czf - -C /logs/agent sessions")
            )
            return code, timed_out
        finally:
            sandbox.terminate()

    return run


def daytona_backend(context: Path, version: str) -> Backend:
    from daytona import (
        CreateSandboxFromImageParams,
        Daytona,
        DaytonaError,
        DaytonaNotFoundError,
        Image,
        Resources,
        SessionExecuteRequest,
    )

    client = Daytona()  # DAYTONA_API_KEY from the environment
    image = Image.from_dockerfile(context / "Dockerfile")

    def run(query_dir: Path, task: Task, env: dict[str, str]) -> tuple[int, bool]:
        sandbox = client.create(
            CreateSandboxFromImageParams(
                image=image,
                os_user="root",  # as the docker and modal backends
                resources=Resources(
                    cpu=max(1, round(task.cpus)),
                    memory=task.memory_mb // 1024,
                    disk=task.storage_mb // 1024,
                ),
                labels={"ff-parity": "original"},
                auto_stop_interval=0,
                ephemeral=True,
                # Reaps a sandbox this process fails to delete.
                ttl_minutes=int(task.timeout // 60) + 60,
            ),
            timeout=1800,
        )
        try:
            session = uuid.uuid4().hex
            sandbox.process.create_session(session)

            def sh(command: str, deadline: float) -> int | None:
                """Exit code, or None if `deadline` (monotonic) passes first."""
                cmd = sandbox.process.execute_session_command(
                    session, SessionExecuteRequest(command=command, run_async=True)
                )
                errors = 0
                while time.monotonic() < deadline:
                    try:
                        code = sandbox.process.get_session_command(
                            session, cmd.cmd_id
                        ).exit_code
                        errors = 0
                    except DaytonaError:
                        # A transient API error must not lose a paid agent run.
                        errors += 1
                        if errors >= 12:
                            raise
                        code = None
                    if code is not None:
                        return int(code)
                    time.sleep(5)
                return None

            sandbox.fs.upload_file(task.instruction.encode(), "/tmp/instruction.md")
            # Variables inline as Harbor's Daytona exec passes them; `timeout`
            # exits 124 when the agent limit is reached.
            env_args = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
            code = sh(
                f"env {env_args} timeout {int(task.timeout)} "
                f"bash -c {shlex.quote(AGENT_COMMAND)} < /tmp/instruction.md",
                time.monotonic() + task.timeout + 300,
            )
            sh(
                "tar -czf /tmp/sessions.tar.gz -C /logs/agent sessions",
                time.monotonic() + 300,
            )
            agent_dir = query_dir / "logs" / "agent"
            for remote, local in (
                ("/app/report.md", query_dir / "app" / "report.md"),
                ("/logs/agent/claude-code.txt", agent_dir / "claude-code.txt"),
                ("/tmp/sessions.tar.gz", agent_dir / "sessions.tar.gz"),
            ):
                try:
                    local.write_bytes(sandbox.fs.download_file(remote) or b"")
                except DaytonaNotFoundError:
                    local.write_bytes(b"")  # missing: no report, or no log
            return (-1, True) if code is None else (code, code == 124)
        finally:
            sandbox.delete()

    return run


BACKENDS = {
    "docker": docker_backend,
    "modal": modal_backend,
    "daytona": daytona_backend,
}


def run_one(backend: Backend, query_dir: Path, task: Task, model: str) -> str:
    status_file = query_dir / "status"
    if status_file.is_file() and status_file.read_text() in FINAL_STATUSES:
        return "skipped"
    for sub in ("app", "logs"):
        shutil.rmtree(query_dir / sub, ignore_errors=True)
    (query_dir / "app").mkdir(parents=True)
    (query_dir / "logs" / "agent").mkdir(parents=True)
    (query_dir / "instruction.md").write_text(task.instruction)
    try:
        code, timed_out = backend(query_dir, task, agent_env(model))
    except Exception as e:  # noqa: BLE001  any backend failure is rerun, not scored
        print(f"{query_dir.name}: backend error: {e!r}", flush=True)
        status_file.write_text("backend_error")
        return "backend_error"
    status = (
        "timeout"
        if timed_out
        else parse_status(query_dir / "logs" / "agent" / "claude-code.txt")
    )
    (query_dir / "exit_code").write_text(str(code))
    status_file.write_text(status)
    return status


def cmd_run(args: argparse.Namespace) -> None:
    if set_vars := [v for v in HOST_OVERRIDES if os.environ.get(v)]:
        raise SystemExit(
            f"unset these first; Harbor's side runs without them: {set_vars}"
        )
    tasks = dataset_tasks(args.dataset)
    if args.limit:
        tasks = dict(list(tasks.items())[: args.limit])
    out = args.out.resolve()
    context = agent_image_dir(args.dataset, args.claude_code_version, out)
    backend = BACKENDS[args.backend](context, args.claude_code_version)
    with ThreadPoolExecutor(args.concurrency) as pool:
        futures = {
            qid: pool.submit(run_one, backend, out / qid, task, args.model)
            for qid, task in tasks.items()
        }
        for qid, fut in futures.items():
            print(f"{qid}: {fut.result()}", flush=True)
    collect([out], None)


def harbor_status(trial: Path) -> str:
    result = (
        json.loads((trial / "result.json").read_text())
        if (trial / "result.json").is_file()
        else {}
    )
    exception = (result.get("exception_info") or {}).get("exception_type", "")
    if "Timeout" in exception and "Agent" in exception:
        return "timeout"
    return parse_status(trial / "agent" / "claude-code.txt")


# status, report, Harbor reward, Harbor grading lost judge votes
Record = tuple[str, str, float | None, bool]


def lost_votes(trial: Path, reward: float | None) -> bool:
    """score()'s regrade rule on a Harbor trial: no reward, or a judge without votes."""
    grading = trial / "verifier" / "grading.json"
    g = json.loads(grading.read_text()) if grading.is_file() else {}
    return (
        reward is None
        or g.get("failure_reason") == "judge_error"
        or bool(g.get("failed_checks_by_judge"))
    )


def records(path: Path) -> dict[str, Record]:
    """query_id -> record from an original-side run dir or a Harbor job dir."""
    found = {}
    for entry in sorted(p for p in path.iterdir() if p.is_dir()):
        config = entry / "config.json"
        if config.is_file():  # Harbor trial
            qid = Path(
                json.loads(config.read_text())["task"]["path"]
            ).name.removeprefix("frontier-finance-")
            report = entry / "artifacts" / "app" / "report.md"
            result = entry / "result.json"
            rewards = (
                (json.loads(result.read_text()).get("verifier_result") or {}).get(
                    "rewards"
                )
                if result.is_file()
                else None
            ) or {}
            reward = rewards.get("reward")
            found[qid] = (
                harbor_status(entry),
                report.read_text() if report.is_file() else "",
                reward,
                lost_votes(entry, reward),
            )
        elif (entry / "instruction.md").is_file():  # original-side query
            status = (
                (entry / "status").read_text()
                if (entry / "status").is_file()
                else "no_result"
            )
            report = entry / "app" / "report.md"
            found[entry.name] = (
                status,
                report.read_text() if report.is_file() else "",
                None,
                False,
            )
    return found


def collect(paths: list[Path], summaries: Path | None) -> None:
    """paths are one run: a first attempt, then reruns that replace its queries."""
    recs: dict[str, Record] = {}
    attempts: Counter[str] = Counter()
    for path in paths:
        for qid, rec in records(path).items():
            attempts[qid] += 1
            recs[qid] = rec
    counts = Counter(status for status, *_ in recs.values())
    rerun = sorted(q for q, (s, *_) in recs.items() if s not in FINAL_STATUSES)
    regrade = sorted(
        q for q, (s, _, _, lost) in recs.items() if s in FINAL_STATUSES and lost
    )
    no_report = sum(1 for _, report, *_ in recs.values() if not report.strip())
    print(
        f"== {len(recs)} queries, {no_report} without a report, "
        f"{sum(n > 1 for n in attempts.values())} rerun at least once  "
        + "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    if rerun:
        print(f"   to rerun ({len(rerun)}): {' '.join(rerun)}")
        print("   Harbor side: harbor run ... " + " ".join(f"-i '*{q}'" for q in rerun))
    if regrade:
        print(
            f"   to regrade, left out of the mean ({len(regrade)}): {' '.join(regrade)}"
        )
    rewards = [
        r
        for s, _, r, lost in recs.values()
        if s in FINAL_STATUSES and r is not None and not lost
    ]
    if rewards:
        print(
            f"   Harbor mean reward over {len(rewards)} final trials: "
            f"{sum(rewards) / len(rewards):.4f}"
        )
    if summaries:
        rows = [
            {"query_id": q, "system_summary": report}
            for q, (_, report, *_) in sorted(recs.items())
        ]
        summaries.parent.mkdir(parents=True, exist_ok=True)
        summaries.write_text(json.dumps(rows, indent=2))
        print(f"   wrote {len(rows)} summaries to {summaries}")


def score(per_item_files: list[Path]) -> float:
    """Upstream macro_avg_qualification_rate_on_all_queries over the merged files.

    Queries whose grading lost any judge's votes (judge_error, or a non-empty
    failed_checks_by_judge) are listed for a regrade and left out until then.
    """
    items = {}
    for path in per_item_files:
        for item in json.loads(path.read_text()):
            items[item["query_id"]] = item
    regrade = sorted(
        q
        for q, it in items.items()
        if it["failure_reason"] == "judge_error" or it["failed_checks_by_judge"]
    )
    rates = [
        it["num_qualified"] / it["num_rubrics"] if it["num_rubrics"] else 0.0
        for q, it in items.items()
        if q not in regrade
    ]
    macro = sum(rates) / len(rates) if rates else 0.0
    print(f"== {len(items)} graded queries, {len(rates)} scored, macro {macro:.4f}")
    if regrade:
        print(f"   to regrade ({len(regrade)}): {' '.join(regrade)}")
    return macro


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run claude-code on every query outside Harbor")
    p.add_argument(
        "-d",
        "--dataset",
        type=Path,
        required=True,
        help="Harbor-generated FrontierFinance tasks",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        help="run directory, one subdirectory per query",
    )
    p.add_argument(
        "-m",
        "--model",
        required=True,
        help="model id, as in Harbor's -m without 'anthropic/'",
    )
    p.add_argument(
        "--claude-code-version",
        required=True,
        help="same as the Harbor agent's version kwarg",
    )
    p.add_argument("-n", "--concurrency", type=int, default=8)
    p.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default="docker",
        help="where each task's container runs",
    )
    p.add_argument("--limit", type=int, help="first N queries only (sanity run)")
    p = sub.add_parser("collect", help="apply the rerun rule, write summaries")
    p.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="one run: an original-side run dir, or a Harbor job dir then its rerun jobs",
    )
    p.add_argument("--summaries", type=Path, help="write system_summaries.json")
    p = sub.add_parser("score", help="merge per_item.json files into the macro rate")
    p.add_argument("per_item", type=Path, nargs="+", help="later files replace earlier")
    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "collect":
        collect(args.paths, args.summaries)
    else:
        score(args.per_item)


if __name__ == "__main__":
    main()
