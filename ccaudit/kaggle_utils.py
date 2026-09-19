"""
ccaudit.kaggle_utils -- everything the session notebooks need from the
Kaggle environment.

The principal job of this module is auto-discovery.  Kaggle mounts an
attached dataset or notebook output under a path that is not under the
notebook's control and cannot be predicted (`/kaggle/input/<slug>/...`,
sometimes with an extra folder level from the zip).  Every session notebook
therefore names *what it needs* (an index.json, a cache file, a checkpoint)
and this module finds it.

Environment facts this module assumes (free tier):
  session wall clock   10 h
  /kaggle/working      19 GB, kept as the notebook's output
  /kaggle/temp         ~55 GB, erased at session end
  GPU                  2 x T4, 15 GB each, no NVLink -> two independent cards
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

INPUT_ROOT = "/kaggle/input"
WORK_ROOT = "/kaggle/working"
TEMP_ROOT = "/kaggle/temp"

OUTPUT_CAP_GB = 19.0
SESSION_WALL_H = 10.0

# Directories never descended into when scanning for inputs.
_SKIP_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints", "node_modules", ".cache",
}


def on_kaggle() -> bool:
    return os.path.isdir(INPUT_ROOT) or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def work_dir(sub: str = "") -> str:
    base = WORK_ROOT if os.path.isdir(WORK_ROOT) else os.path.abspath("./work")
    p = os.path.join(base, sub) if sub else base
    os.makedirs(p, exist_ok=True)
    return p


def temp_dir(sub: str = "") -> str:
    for base in (TEMP_ROOT, "/tmp"):
        try:
            p = os.path.join(base, "ccaudit", sub) if sub else os.path.join(base, "ccaudit")
            os.makedirs(p, exist_ok=True)
            return p
        except OSError:
            continue
    p = os.path.abspath("./tmp")
    os.makedirs(p, exist_ok=True)
    return p


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------

def _walk(root: str, max_depth: int = 8) -> Iterable[Tuple[str, List[str], List[str]]]:
    root = os.path.abspath(root)
    base_depth = root.rstrip("/").count("/")
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        if dirpath.rstrip("/").count("/") - base_depth >= max_depth:
            dirnames[:] = []
        yield dirpath, dirnames, filenames


def find_input_files(
    filename: str,
    under: Optional[str] = None,
    roots: Optional[Sequence[str]] = None,
    max_depth: int = 8,
) -> List[str]:
    """
    All files named `filename` anywhere under the given roots (default: every
    attached dataset plus /kaggle/working).  `under` optionally requires that
    the parent directory be named that, e.g. find_input_files("index.json",
    under="parsed").  Results are sorted: /kaggle/working first (a run in
    progress takes precedence over a stale attached copy), then
    alphabetically.
    """
    roots = list(roots) if roots else [WORK_ROOT, INPUT_ROOT]
    hits: List[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _d, files in _walk(root, max_depth):
            if filename in files:
                if under and os.path.basename(dirpath) != under:
                    continue
                hits.append(os.path.join(dirpath, filename))

    def rank(p: str) -> Tuple[int, str]:
        return (0 if p.startswith(WORK_ROOT) else 1, p)

    return sorted(set(hits), key=rank)


def find_input_file(filename: str, under: Optional[str] = None, **kw) -> Optional[str]:
    hits = find_input_files(filename, under=under, **kw)
    return hits[0] if hits else None


def find_glob(pattern: str, roots: Optional[Sequence[str]] = None,
              max_depth: int = 8) -> List[str]:
    """All files matching a fnmatch pattern under the roots."""
    import fnmatch

    roots = list(roots) if roots else [WORK_ROOT, INPUT_ROOT]
    hits: List[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _d, files in _walk(root, max_depth):
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    hits.append(os.path.join(dirpath, f))
    return sorted(set(hits), key=lambda p: (0 if p.startswith(WORK_ROOT) else 1, p))


def find_code_dir() -> str:
    """
    Locate the directory that CONTAINS the `ccaudit` package (so it can be
    put on sys.path).  Looks for ccaudit/kaggle_utils.py as the marker file.
    """
    # Already importable (e.g. running outside Kaggle from a checkout).
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isfile(os.path.join(here, "ccaudit", "kaggle_utils.py")):
        return here
    for hit in find_input_files("kaggle_utils.py", under="ccaudit"):
        return os.path.dirname(os.path.dirname(hit))
    raise FileNotFoundError(
        "Could not find the ccaudit code under /kaggle/input.\n"
        "Add Input -> your code dataset (<your-code-dataset>). If it is "
        "attached, it may be an older version without ccaudit/kaggle_utils.py, "
        "or the zip nested an extra folder -- open the dataset preview and "
        "check that ccaudit/kaggle_utils.py is visible."
    )


def install_code() -> str:
    """find_code_dir() and put it on sys.path.  Returns the directory."""
    d = find_code_dir()
    if d not in sys.path:
        sys.path.insert(0, d)
    return d


def find_dataset_root(markers: Sequence[str] = ("original",)) -> Optional[str]:
    """
    Locate the FaceForensics++ mirror: the directory that directly contains an
    `original/` folder (and, normally, the manipulation folders next to it).
    """
    best: Optional[str] = None
    for root in (INPUT_ROOT,):
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _f in _walk(root, max_depth=6):
            if all(m in dirnames for m in markers):
                n_sub = len([d for d in dirnames if not d.startswith(".")])
                if best is None or n_sub > len(os.listdir(best)):
                    best = dirpath
    return best


def find_celebdf_root() -> Optional[str]:
    """Directory containing Celeb-real/ and Celeb-synthesis/."""
    for root in (INPUT_ROOT,):
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _f in _walk(root, max_depth=6):
            if "Celeb-synthesis" in dirnames and (
                "Celeb-real" in dirnames or "YouTube-real" in dirnames
            ):
                return dirpath
    return None


def find_parsed_index() -> Optional[str]:
    """The parsed/index.json produced by session 1 (or a merged one)."""
    hits = find_input_files("index.json", under="parsed")
    return hits[0] if hits else None


def find_all_parsed_indices() -> List[str]:
    return find_input_files("index.json", under="parsed")


def find_run_dirs(prefix: str = "raw_") -> List[str]:
    """
    Every directory that contains raw_*.json files, i.e. every run folder
    from any attached session output.  Used by the final report session.
    """
    dirs = {os.path.dirname(p) for p in find_glob(prefix + "*.json")}
    return sorted(dirs, key=lambda p: (0 if p.startswith(WORK_ROOT) else 1, p))


def find_caches(pattern: str = "cache_*.json") -> List[str]:
    return find_glob(pattern)


def find_landmarker_task() -> Optional[str]:
    """A manually uploaded face_landmarker.task, if one is attached."""
    return find_input_file("face_landmarker.task")


def find_checkpoints(pattern: str = "*.pt") -> List[str]:
    return find_glob(pattern)


# --------------------------------------------------------------------------
# shell
# --------------------------------------------------------------------------


def package_root() -> str:
    """Directory containing the `ccaudit` package (this file's grandparent)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _child_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Environment for a child process.

    A subprocess does not inherit the parent's sys.path, so
    `python -m ccaudit.<module>` fails with ModuleNotFoundError unless the
    code directory is on PYTHONPATH.  On Kaggle the code lives under
    /kaggle/input/<slug>/, which is never on the default path.  Every command
    this module launches therefore gets PYTHONPATH prepended automatically.
    """
    e = dict(os.environ)
    root = package_root()
    prev = e.get("PYTHONPATH", "")
    if root not in prev.split(os.pathsep):
        e["PYTHONPATH"] = root + (os.pathsep + prev if prev else "")
    if extra:
        e.update({k: str(v) for k, v in extra.items()})
    return e


def sh(
    cmd: str | Sequence[str],
    check: bool = True,
    env: Optional[Dict[str, str]] = None,
    log: Optional[str] = None,
    echo: bool = True,
    cwd: Optional[str] = None,
) -> int:
    """
    Run a command, streaming its output to the notebook (and optionally to a
    log file).  Returns the exit code.  Raises on non-zero if check=True.
    """
    if isinstance(cmd, (list, tuple)):
        cmd_s = " ".join(shlex.quote(str(c)) for c in cmd)
    else:
        cmd_s = cmd
    if echo:
        print(f"$ {cmd_s}", flush=True)
    e = _child_env(env)
    fh = None
    if log:
        os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
        fh = open(log, "a", encoding="utf-8", buffering=1)
        fh.write(f"\n$ {cmd_s}\n")
    t0 = time.time()
    proc = subprocess.Popen(
        cmd_s, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=e, cwd=cwd,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if fh:
            fh.write(line)
    rc = proc.wait()
    if fh:
        fh.write(f"[exit {rc} after {time.time()-t0:.0f}s]\n")
        fh.close()
    if echo:
        print(f"[exit {rc} after {(time.time()-t0)/60:.1f} min]", flush=True)
    if check and rc != 0:
        raise RuntimeError(f"command failed with exit code {rc}: {cmd_s}")
    return rc


def run_parallel(
    cmds: Sequence[str],
    envs: Optional[Sequence[Dict[str, str]]] = None,
    logs: Optional[Sequence[str]] = None,
    check: bool = True,
    poll_sec: int = 30,
) -> List[int]:
    """
    Run several commands concurrently, each with its own environment and log
    file.  Used to put one process on each GPU (CUDA_VISIBLE_DEVICES=i) and to
    fan CPU work out over the available cores.  Output goes to the log files
    only; the notebook gets a periodic progress line, because interleaved
    stdout from N processes is unreadable.
    """
    procs = []
    handles = []
    for i, cmd in enumerate(cmds):
        e = _child_env(envs[i] if (envs and i < len(envs)) else None)
        log = logs[i] if logs and i < len(logs) else None
        if log:
            os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
            fh = open(log, "a", encoding="utf-8", buffering=1)
            fh.write(f"\n$ {cmd}\n")
        else:
            fh = subprocess.DEVNULL if False else open(os.devnull, "w")
        handles.append(fh)
        print(f"[proc {i}] $ {cmd}" + (f"  -> {log}" if log else ""), flush=True)
        procs.append(subprocess.Popen(
            cmd, shell=True, stdout=fh, stderr=subprocess.STDOUT, text=True, env=e
        ))
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(poll_sec)
        alive = sum(1 for p in procs if p.poll() is None)
        print(f"    [{(time.time()-t0)/60:6.1f} min] {alive}/{len(procs)} running",
              flush=True)
    rcs = [p.wait() for p in procs]
    for fh in handles:
        try:
            fh.close()
        except Exception:
            pass
    print(f"[parallel done in {(time.time()-t0)/60:.1f} min] exit codes: {rcs}",
          flush=True)
    if check and any(rc != 0 for rc in rcs):
        raise RuntimeError(f"a parallel command failed: {rcs}")
    return rcs


def tail(path: str, n: int = 40) -> None:
    if not os.path.exists(path):
        print(f"(no log at {path})")
        return
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    print("".join(lines[-n:]))


def pip_install(spec: str, quiet: bool = True, upgrade: bool = False) -> int:
    flags = "-q " if quiet else ""
    flags += "-U " if upgrade else ""
    return sh(f"{sys.executable} -m pip install {flags}{spec}", check=False)


# --------------------------------------------------------------------------
# staging and reporting
# --------------------------------------------------------------------------

def stage(src: str, dst: str, what: str = "") -> str:
    """
    Copy a read-only input into a writable location.  Kaggle mounts inputs
    read-only, so anything a module needs to append to (a cache, an index it
    will extend) must be staged first.
    """
    import shutil

    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(src, dst)
    print(f"staged {what or os.path.basename(src)}: {src} -> {dst}", flush=True)
    return dst


def disk_report(path: str = WORK_ROOT) -> Dict[str, float]:
    from .common import dir_size_gb, disk_free_gb

    used = dir_size_gb(path) if os.path.isdir(path) else 0.0
    free = disk_free_gb(path)
    tmp = temp_dir()
    tmp_free = disk_free_gb(tmp)
    print(
        f"output {path}: {used:.2f} GB used of {OUTPUT_CAP_GB:.0f} GB cap "
        f"({100*used/OUTPUT_CAP_GB:.0f}%)   |   filesystem free {free:.1f} GB"
        f"   |   scratch {tmp} free {tmp_free:.1f} GB",
        flush=True,
    )
    if used > 0.85 * OUTPUT_CAP_GB:
        print("!! WARNING: close to the 19 GB output cap. Move big files to "
              f"{tmp} (scratch, erased at session end).", flush=True)
    return {"used_gb": used, "free_gb": free, "tmp_free_gb": tmp_free}


def gpu_report() -> List[Dict[str, str]]:
    from .common import gpu_info

    gpus = gpu_info()
    if not gpus:
        print("no GPU visible (CPU session)", flush=True)
    for i, g in enumerate(gpus):
        print(f"GPU {i}: {g['name']}  {g['memory']}  driver {g.get('driver','')}",
              flush=True)
    return gpus


def n_gpus() -> int:
    from .common import gpu_info

    return len(gpu_info())


def write_env_provenance(out_dir: str) -> None:
    """pip freeze and nvidia-smi output into audit/logs/ (one per session)."""
    logs = os.path.join(out_dir, "logs")
    os.makedirs(logs, exist_ok=True)
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=180,
        ).stdout
        with open(os.path.join(logs, "pip_freeze.txt"), "w") as fh:
            fh.write(freeze)
    except Exception as exc:
        print(f"(pip freeze failed: {exc})")
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                             timeout=30).stdout
    except Exception:
        smi = "no nvidia-smi\n"
    with open(os.path.join(logs, "gpu.txt"), "w") as fh:
        fh.write(smi)


def session_header(name: str, out_dir: str) -> None:
    """Standard first-cell output: what is attached, what was found, limits."""
    from .common import banner

    print(banner(f"audit session: {name}"))
    print(f"python {sys.version.split()[0]}   kaggle={on_kaggle()}")
    if os.path.isdir(INPUT_ROOT):
        print("\nattached inputs:")
        for d in sorted(os.listdir(INPUT_ROOT)):
            p = os.path.join(INPUT_ROOT, d)
            if os.path.isdir(p):
                try:
                    sub = sorted(os.listdir(p))[:6]
                except OSError:
                    sub = []
                print(f"  /kaggle/input/{d}   {sub}")
    print()
    gpu_report()
    disk_report()
    os.makedirs(out_dir, exist_ok=True)
    write_env_provenance(out_dir)
    print(f"\noutput dir: {out_dir}")
    print(f"scratch   : {temp_dir()}")
    print(f"session wall clock: {SESSION_WALL_H} h -- every long step below "
          f"carries a hard --time-budget-min well under it.\n", flush=True)
