# Agent environment for the original side of Harbor parity. It mirrors the
# Harbor task image (python:3.13-slim with git and curl, WORKDIR /app) plus what
# Harbor's claude-code agent installs at run time (nodejs, npm, procps, and the
# Claude Code bootstrap at a pinned version).
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates bash nodejs npm procps \
    && rm -rf /var/lib/apt/lists/*

ARG CLAUDE_CODE_VERSION
RUN test -n "$CLAUDE_CODE_VERSION" \
    && curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh | bash -s -- "$CLAUDE_CODE_VERSION" \
    && /root/.local/bin/claude --version
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
