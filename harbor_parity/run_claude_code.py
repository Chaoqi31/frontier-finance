"""Original side of Harbor parity for FrontierFinance: claude-code outside Harbor.

  run      Run claude-code once per query in a fresh container built from
           harbor_parity/Dockerfile, on local Docker or in a Modal sandbox
           (--backend), with the same instruction text, model, claude-code
           version, flags, environment, resources and time limit as Harbor's
           claude-code agent. Rerunning the same command skips queries that
           already have a final status and retries the rest.
  collect  Count runs by status and write system_summaries.json for the grader.
           Reads an original-side run directory or a Harbor job directory, so
           both sides are counted, and the Harbor side can be cross-graded here.

Usage (from the repository root). ANTHROPIC_API_KEY, or ANTHROPIC_AUTH_TOKEN for a
bearer-auth gateway, and an optional ANTHROPIC_BASE_URL come from the environment;
--backend modal also needs the modal package and a Modal login:

  python harbor_parity/run_claude_code.py run -d <harbor>/datasets/frontier-finance \\
      -o runs/original1 -m claude-haiku-5-5 --claude-code-version 2.1.326 -n 8
  python harbor_parity/run_claude_code.py collect runs/original1 \\
      --summaries runs/original1/system_summaries.json

Then grade with this repository's grader and eval.example.yaml unchanged
(responses_path -> the summaries).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
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

HERE = Path(__file__).parent
# Statuses that are real results. Anything else (an API error, no result event)
# is an infrastructure failure and is rerun on both sides.
FINAL_STATUSES = frozenset({"success", "error_max_turns", "timeout"})
ALIASES = ("OPUS", "SONNET", "HAIKU", "FABLE")
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
        )
    return tasks


def agent_env(model: str) -> dict[str, str]:
    """The variables Harbor's claude-code agent sets (harbor 0.26.0, claude_code.py run()), plus the keys."""
    env = {
        "ANTHROPIC_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "IS_SANDBOX": "1",
        "CLAUDE_CONFIG_DIR": "/logs/agent/sessions",
    }
    if os.environ.get("ANTHROPIC_BASE_URL"):
        env.update({f"ANTHROPIC_DEFAULT_{tier}_MODEL": model for tier in ALIASES})
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


def docker_backend(version: str) -> Backend:
    image = f"ff-parity-claude-code:{version}"
    subprocess.run(
        [
            "docker",
            "build",
            "-q",
            "-t",
            image,
            "--build-arg",
            f"CLAUDE_CODE_VERSION={version}",
            str(HERE),
        ],
        check=True,
    )

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


def modal_backend(version: str) -> Backend:
    import modal

    image = modal.Image.from_dockerfile(
        HERE / "Dockerfile",
        context_dir=HERE,
        build_args={"CLAUDE_CODE_VERSION": version},
    )
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


BACKENDS = {"docker": docker_backend, "modal": modal_backend}


def run_one(backend: Backend, query_dir: Path, task: Task, model: str) -> str:
    status_file = query_dir / "status"
    if status_file.is_file() and status_file.read_text() in FINAL_STATUSES:
        return "skipped"
    for sub in ("app", "logs"):
        shutil.rmtree(query_dir / sub, ignore_errors=True)
    (query_dir / "app").mkdir(parents=True)
    (query_dir / "logs" / "agent").mkdir(parents=True)
    (query_dir / "instruction.md").write_text(task.instruction)
    code, timed_out = backend(query_dir, task, agent_env(model))
    status = (
        "timeout"
        if timed_out
        else parse_status(query_dir / "logs" / "agent" / "claude-code.txt")
    )
    (query_dir / "exit_code").write_text(str(code))
    status_file.write_text(status)
    return status


def cmd_run(args: argparse.Namespace) -> None:
    tasks = dataset_tasks(args.dataset)
    if args.limit:
        tasks = dict(list(tasks.items())[: args.limit])
    backend = BACKENDS[args.backend](args.claude_code_version)
    out = args.out.resolve()
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


def records(path: Path) -> dict[str, tuple[str, str]]:
    """query_id -> (status, report) from an original-side run dir or a Harbor job dir."""
    found = {}
    for entry in sorted(p for p in path.iterdir() if p.is_dir()):
        config = entry / "config.json"
        if config.is_file():  # Harbor trial
            qid = Path(
                json.loads(config.read_text())["task"]["path"]
            ).name.removeprefix("frontier-finance-")
            report = entry / "artifacts" / "app" / "report.md"
            found[qid] = (
                harbor_status(entry),
                report.read_text() if report.is_file() else "",
            )
        elif (entry / "instruction.md").is_file():  # original-side query
            status = (
                (entry / "status").read_text()
                if (entry / "status").is_file()
                else "no_result"
            )
            report = entry / "app" / "report.md"
            found[entry.name] = (status, report.read_text() if report.is_file() else "")
    return found


def collect(paths: list[Path], summaries: Path | None) -> None:
    for path in paths:
        recs = records(path)
        counts = Counter(status for status, _ in recs.values())
        rerun = sorted(q for q, (s, _) in recs.items() if s not in FINAL_STATUSES)
        no_report = sum(1 for _, report in recs.values() if not report.strip())
        print(
            f"== {path}: {len(recs)} queries, {no_report} without a report  "
            + "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        )
        if rerun:
            print(f"   to rerun ({len(rerun)}): {' '.join(rerun)}")
        if summaries:
            rows = [
                {"query_id": q, "system_summary": report}
                for q, (_, report) in sorted(recs.items())
            ]
            summaries.parent.mkdir(parents=True, exist_ok=True)
            summaries.write_text(json.dumps(rows, indent=2))
            print(f"   wrote {len(rows)} summaries to {summaries}")


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
        help="Anthropic model id, as in Harbor's -m without 'anthropic/'",
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
    p = sub.add_parser("collect", help="count statuses and write system_summaries.json")
    p.add_argument(
        "paths", type=Path, nargs="+", help="original-side run dirs or Harbor job dirs"
    )
    p.add_argument(
        "--summaries", type=Path, help="write system_summaries.json (one path only)"
    )
    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    else:
        if args.summaries and len(args.paths) != 1:
            parser.error("--summaries takes exactly one path")
        collect(args.paths, args.summaries)


if __name__ == "__main__":
    main()
