# -*- coding: utf-8 -*-
"""
Best-effort isolation for model-generated Python scripts (dynamic_script path).

Every generated solver, sweep variant and validation test script is started through
:func:`run_generated_script`, which

* runs the script in a fresh temporary working directory (never the repository),
* passes a minimal environment: no API keys, tokens, cloud credentials or
  ``VECTORNAUT_*`` settings; HOME/TMP point into the temporary directory,
* points HTTP(S) proxies at an unreachable local port (best effort only),
* starts the script in its own process group / session and kills the whole group
  (including grandchildren) on timeout or when the script returns,
* on POSIX limits CPU time, address space and file size via ``resource`` (set by a
  tiny trusted launcher that then ``exec``s the interpreter, so no ``preexec_fn`` is
  needed in a multi-threaded server).

:func:`check_generated_code` is a static pre-check that rejects scripts importing
modules or calling functions a numerical script does not need (network, process
spawning, ctypes, recursive deletes). A rejection is fed into the generators'
self-correction loop.

This is NOT a security boundary. A determined script can still read files the user
can read, open network connections through modules not on the deny list or via
native extensions, or escape the process group by calling setsid from native code.
Run untrusted or public workloads only inside a container / VM (see
docs/OPERATIONS.md, "Generated-script sandbox").
"""
import ast
import contextlib
import glob
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence

IS_WINDOWS = os.name == "nt"

# --- limits (overridable via environment) -----------------------------------------
DEFAULT_MEMORY_MB = 2048
DEFAULT_FILE_SIZE_MB = 200
# CPU seconds granted on top of the wall-clock timeout (RLIMIT_CPU soft limit).
CPU_LIMIT_MARGIN_SECONDS = 5

# Unroutable local port for the proxy variables (port 9 = "discard").
_DEAD_PROXY = "http://127.0.0.1:9"

# Environment variables copied from the parent (only if set). Everything else is dropped.
_PASSTHROUGH_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ")
# Windows needs these for the interpreter itself (crypto RNG, temp dirs, DLL lookup).
_WINDOWS_PASSTHROUGH_ENV = ("SYSTEMROOT", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE")
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "GOOGLE", "GEMINI", "AWS", "AZURE")

# --- static pre-check -----------------------------------------------------------------
# Top-level modules a numerical solver or test script never needs.
FORBIDDEN_MODULES = frozenset({
    "socket", "socketserver", "ssl", "select", "selectors",
    "requests", "urllib", "urllib2", "urllib3", "http", "httpx", "aiohttp", "websocket", "websockets",
    "ftplib", "smtplib", "poplib", "imaplib", "telnetlib", "nntplib", "xmlrpc", "webbrowser", "paramiko",
    "ctypes", "cffi",
    "subprocess", "pty", "pexpect",
    "importlib", "builtins",
})
# Attributes of os / shutil that spawn processes, send signals, leave the process group
# or delete directory trees.
FORBIDDEN_ATTRIBUTES = {
    "os": frozenset({
        "system", "popen", "fork", "forkpty", "kill", "killpg", "setsid", "setpgid", "setpgrp",
        "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
        "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
        "posix_spawn", "posix_spawnp", "startfile", "removedirs",
    }),
    "shutil": frozenset({"rmtree"}),
}
FORBIDDEN_BUILTIN_CALLS = frozenset({"exec", "eval", "compile", "__import__"})


@dataclass
class SandboxResult:
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float
    workdir: str

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.returncode == 0

    def error_text(self, limit: int = 8000) -> str:
        """stderr (or stdout) tail for error messages / correction prompts."""
        text = self.stderr or self.stdout or ""
        if self.timed_out:
            text = f"Timed out after {self.duration_s:.0f} s; the process group was killed.\n{text}"
        return text[-limit:]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def script_timeout_seconds() -> int:
    return max(1, _env_int("VECTORNAUT_SCRIPT_TIMEOUT_SECONDS", 60))


def _looks_secret(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _SECRET_MARKERS)


def _mpl_cache_dir() -> str:
    return os.path.join(tempfile.gettempdir(), "vectornaut_sandbox_mplcache")


def sandbox_env(workdir: str, extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Minimal environment for a generated script running in ``workdir``."""
    home = os.path.join(workdir, ".home")
    tmp = os.path.join(workdir, ".tmp")
    mpl = os.path.join(home, ".matplotlib")
    for path in (home, tmp, mpl):
        os.makedirs(path, exist_ok=True)

    names = list(_PASSTHROUGH_ENV) + (list(_WINDOWS_PASSTHROUGH_ENV) if IS_WINDOWS else [])
    env = {name: os.environ[name] for name in names if name in os.environ and not _looks_secret(name)}
    env.setdefault("PATH", os.defpath)
    env.setdefault("LANG", "C.UTF-8")
    env.update({
        "HOME": home,
        "USERPROFILE": home,
        "TMPDIR": tmp,
        "TEMP": tmp,
        "TMP": tmp,
        "MPLCONFIGDIR": mpl,
        "MPLBACKEND": "Agg",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        # Single-threaded BLAS: keeps the address-space limit meaningful and CPU use bounded.
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        # Best-effort network deterrent: HTTP clients that honour proxy variables fail fast.
        "HTTP_PROXY": _DEAD_PROXY,
        "HTTPS_PROXY": _DEAD_PROXY,
        "ALL_PROXY": _DEAD_PROXY,
        "http_proxy": _DEAD_PROXY,
        "https_proxy": _DEAD_PROXY,
        "all_proxy": _DEAD_PROXY,
        "NO_PROXY": "",
        "no_proxy": "",
    })
    for key, value in (extra_env or {}).items():
        if _looks_secret(key):
            raise ValueError(f"Refusing to pass secret-looking variable {key!r} to a generated script.")
        env[key] = str(value)
    return env


# Trusted launcher: applies resource limits in the child, then execs the interpreter.
# Avoids preexec_fn, which is unsafe in multi-threaded parents (uvicorn workers).
_POSIX_LAUNCHER = r"""
import os, sys
try:
    import resource
except ImportError:
    resource = None
cpu, mem_mb, fsize_mb = (int(v) for v in sys.argv[1:4])
def _limit(name, soft, hard=None):
    if resource is None or not hasattr(resource, name):
        return
    try:
        resource.setrlimit(getattr(resource, name), (soft, soft if hard is None else hard))
    except (ValueError, OSError):
        pass
if cpu > 0:
    _limit("RLIMIT_CPU", cpu, cpu + 5)
if mem_mb > 0:
    _limit("RLIMIT_AS", mem_mb * 1024 * 1024)
if fsize_mb > 0:
    _limit("RLIMIT_FSIZE", fsize_mb * 1024 * 1024)
_limit("RLIMIT_CORE", 0)
os.execv(sys.argv[5], sys.argv[5:])
"""


def _command(args: Sequence[str], timeout: float) -> List[str]:
    python = [sys.executable, "-X", "utf8"]
    if IS_WINDOWS:
        return python + list(args)
    cpu = int(timeout) + CPU_LIMIT_MARGIN_SECONDS
    mem = _env_int("VECTORNAUT_SANDBOX_MEMORY_MB", DEFAULT_MEMORY_MB)
    fsize = _env_int("VECTORNAUT_SANDBOX_FILE_SIZE_MB", DEFAULT_FILE_SIZE_MB)
    return [sys.executable, "-c", _POSIX_LAUNCHER, str(cpu), str(mem), str(fsize), "--"] + python + list(args)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the process and everything in its process group (POSIX) / tree (Windows)."""
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
            )
        except Exception:
            pass
        with contextlib.suppress(Exception):
            proc.kill()
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(Exception):
        proc.kill()


def _seed_mpl_cache(mpl_dir: str) -> None:
    # Building matplotlib's font cache takes seconds; copy a previously built one in.
    for path in glob.glob(os.path.join(_mpl_cache_dir(), "fontlist-*.json")):
        with contextlib.suppress(OSError):
            shutil.copy2(path, mpl_dir)


def _store_mpl_cache(mpl_dir: str) -> None:
    cache = _mpl_cache_dir()
    if glob.glob(os.path.join(cache, "fontlist-*.json")):
        return
    built = glob.glob(os.path.join(mpl_dir, "fontlist-*.json"))
    if built:
        with contextlib.suppress(OSError):
            os.makedirs(cache, exist_ok=True)
            for path in built:
                shutil.copy2(path, cache)


def run_generated_script(
    args: Sequence[str],
    timeout: Optional[float] = None,
    workdir: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> SandboxResult:
    """
    Run ``python <args...>`` (``args[0]`` is the script, paths relative to ``workdir``)
    with the isolation described in the module docstring. ``workdir`` must be a fresh
    directory outside the repository (see :func:`sandbox_workdir`); inputs have to be
    copied into it beforehand and outputs copied out afterwards.
    """
    timeout = float(timeout if timeout is not None else script_timeout_seconds())
    owns_workdir = workdir is None
    if owns_workdir:
        workdir = tempfile.mkdtemp(prefix="vectornaut_sbx_")
    workdir = os.path.abspath(workdir)
    env = sandbox_env(workdir, extra_env)
    _seed_mpl_cache(env["MPLCONFIGDIR"])

    popen_kwargs = dict(
        cwd=workdir,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
    )
    if IS_WINDOWS:
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    start = time.monotonic()
    proc = subprocess.Popen(_command(args, timeout), **popen_kwargs)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
    finally:
        # Also reap stragglers (background grandchildren) after a normal exit.
        if not IS_WINDOWS:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(proc.pid, signal.SIGKILL)
    duration = time.monotonic() - start
    _store_mpl_cache(env["MPLCONFIGDIR"])
    return SandboxResult(
        returncode=None if timed_out else proc.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
        timed_out=timed_out,
        duration_s=duration,
        workdir=workdir,
    )


@contextlib.contextmanager
def sandbox_workdir(prefix: str = "vectornaut_sbx_") -> Iterator[str]:
    """Fresh temporary working directory (outside the repository), removed afterwards."""
    base = os.environ.get("VECTORNAUT_SANDBOX_TMPDIR") or None
    path = tempfile.mkdtemp(prefix=prefix, dir=base)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def copy_if_exists(src: str, dst: str) -> bool:
    if not os.path.isfile(src):
        return False
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    shutil.copyfile(src, dst)
    return True


# --- static pre-check -----------------------------------------------------------------

def check_generated_code(code: str) -> List[str]:
    """
    Return a list of human-readable violations (empty when the code is acceptable).
    Syntax errors are not reported here; running the script surfaces them with a
    proper traceback.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    violations: List[str] = []
    aliases: Dict[str, str] = {}  # local name -> module name (os, shutil)

    def flag(node: ast.AST, message: str) -> None:
        violations.append(f"line {getattr(node, 'lineno', '?')}: {message}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in FORBIDDEN_MODULES:
                    flag(node, f"import of forbidden module '{alias.name}'")
                if top in FORBIDDEN_ATTRIBUTES:
                    aliases[alias.asname or top] = top
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0]
            if top in FORBIDDEN_MODULES:
                flag(node, f"import from forbidden module '{module}'")
            elif top in FORBIDDEN_ATTRIBUTES:
                for alias in node.names:
                    if alias.name in FORBIDDEN_ATTRIBUTES[top] or alias.name == "*":
                        flag(node, f"import of forbidden function '{module}.{alias.name}'")

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = aliases.get(node.value.id)
            if module and node.attr in FORBIDDEN_ATTRIBUTES[module]:
                flag(node, f"use of forbidden function '{module}.{node.attr}'")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_BUILTIN_CALLS:
                flag(node, f"call of forbidden builtin '{func.id}()'")
            elif isinstance(func, ast.Name) and func.id == "getattr" and len(node.args) >= 2:
                target, attr = node.args[0], node.args[1]
                if (isinstance(target, ast.Name) and target.id in aliases
                        and isinstance(attr, ast.Constant) and isinstance(attr.value, str)
                        and attr.value in FORBIDDEN_ATTRIBUTES[aliases[target.id]]):
                    flag(node, f"use of forbidden function '{aliases[target.id]}.{attr.value}' via getattr")
    return violations


SANDBOX_RULES_TEXT = (
    "Generated scripts run in a restricted sandbox: temporary working directory, no credentials in "
    "the environment, no network, CPU/memory/file-size limits. Scripts must not import "
    + ", ".join(sorted(FORBIDDEN_MODULES))
    + ", and must not call os.system/os.popen/os.exec*/os.spawn*/os.fork/os.kill/os.setsid, "
    "shutil.rmtree, exec(), eval(), compile() or __import__()."
)


def format_rejection(violations: Sequence[str]) -> str:
    lines = "\n".join(f"- {v}" for v in violations)
    return (
        "The script was REJECTED by the static sandbox pre-check and was not executed:\n"
        f"{lines}\n\n{SANDBOX_RULES_TEXT}"
    )
