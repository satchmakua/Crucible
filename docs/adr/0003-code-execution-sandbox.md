# 3. Code-execution sandbox: guardrail, opt-in, not a jail

- **Status:** Accepted
- **Date:** 2026-06-28

## Context

The code track (M5) verifies solutions by **running model-generated code** against unit
tests — ground truth for code, but a real hazard: the code is untrusted and the harness
often runs on a Windows workstation with no container. DESIGN §4.4/§9 require: locked-down
subprocess, hard timeout, no network, scratch temp dir, WSL2/Docker when available, and
an opt-in flag. A true sandbox (seccomp, namespaces) isn't portable to Windows without
containers, so we must be honest about what the in-process guards do and don't cover.

## Decision

Execution is **off by default** and only runs when the caller passes
`allow_code_execution` (`--allow-code-exec`); the runner refuses code datasets otherwise,
*before* any code is generated or run. When enabled, `verify/code_sandbox.run_in_sandbox`:

- runs the candidate + tests in a **fresh subprocess** (`sys.executable -I`, isolated
  mode: ignores env/user-site, doesn't put cwd on `sys.path`) — never in-process;
- enforces a **hard wall-clock timeout** (process killed on expiry);
- uses a **scratch temp dir** as cwd, deleted afterward, with `TEMP`/`TMP` pinned to it;
- **scrubs the environment** (drops `*_PROXY` vars);
- injects a preamble that **disables network** (neuters `socket.socket` /
  `create_connection`) and **caps CPU time** on POSIX (`RLIMIT_CPU`).

The verifier returns pass/fail from the subprocess exit code; a clean exit means all
asserts/`check()` passed.

## Consequences

- The harness process is protected from hangs and crashes, and casual network/CPU abuse
  is blocked — enough for evaluating benchmark solutions (HumanEval/MBPP) on a dev box.
- **This is a guardrail, not a jail.** The network/CPU guards are in-Python and can be
  bypassed by native extensions or `os`-level calls; grandchild processes may outlive the
  timeout on Windows. For genuinely untrusted code at scale, run the whole harness inside
  **Docker `--network none`** or **WSL2** with namespaces/`firejail` — the opt-in flag and
  this ADR make that boundary explicit.
- The opt-in gate is tested (code datasets error without the flag), and the sandbox is
  tested against correct/wrong/raising/timing-out/network-touching snippets.
