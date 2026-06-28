"""Run model-generated code against unit tests in a locked-down subprocess.

**Threat model (read `docs/adr/0003-…`).** This is a *guardrail*, not a true jail. The
candidate runs in a fresh, isolated Python subprocess with:

- a **hard wall-clock timeout** (the process is killed on expiry);
- a **scratch temp dir** as cwd, deleted afterward;
- **isolated mode** (`python -I`: ignores env/user-site, doesn't add cwd to `sys.path`);
- a **scrubbed environment** (proxy vars dropped, TEMP pinned to the scratch dir);
- an injected preamble that **disables network** (neuters `socket`) and **caps CPU**
  time on POSIX.

It never runs code in-process. For genuinely untrusted code at scale, run the whole
harness inside Docker `--network none` or WSL2 — the in-Python guards can be bypassed by
native extensions. Execution is **opt-in** (`allow_code_execution`); nothing here runs
unless the caller has explicitly enabled it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_PREAMBLE = """\
# --- Crucible sandbox preamble (injected) ---
try:
    import resource as _resource
    _resource.setrlimit(_resource.RLIMIT_CPU, ({cpu}, {cpu}))
except Exception:
    pass
try:
    import socket as _socket
    def _no_network(*_a, **_k):
        raise RuntimeError("network access is disabled in the Crucible sandbox")
    _socket.socket = _no_network
    _socket.create_connection = _no_network
except Exception:
    pass
# --- end preamble ---
"""


@dataclass(frozen=True)
class SandboxResult:
    passed: bool
    detail: str


def run_in_sandbox(code: str, tests: list[str], *, timeout: float = 10.0) -> SandboxResult:
    """Execute `code` + `tests` in an isolated subprocess; pass iff it exits cleanly."""
    cpu = max(1, int(timeout) + 1)
    script = _PREAMBLE.format(cpu=cpu) + "\n" + code + "\n\n" + "\n".join(tests) + "\n"

    env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}

    with tempfile.TemporaryDirectory(prefix="crucible-sbx-") as tmp:
        script_path = Path(tmp) / "candidate.py"
        script_path.write_text(script, encoding="utf-8")
        env["TMP"] = env["TEMP"] = tmp
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(passed=False, detail=f"timed out after {timeout:g}s")

        if proc.returncode == 0:
            return SandboxResult(passed=True, detail="all tests passed")
        lines = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = lines[-1].strip() if lines else f"exited with code {proc.returncode}"
        return SandboxResult(passed=False, detail=detail[:200])
