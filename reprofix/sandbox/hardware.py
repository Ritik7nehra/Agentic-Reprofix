"""What hardware did the run see? Recorded from `nvidia-smi`, with its raw output, so the report can show the proof.

The probe asks the same place the repository's code runs: inside the sandbox container started with the run phase's flags
(`--gpus all`, no network) for the Docker backend, on the host for the development backend. "A GPU is visible there" is all it
shows. It does not show that the repository's code used the GPU, and nothing here has been run on GPU hardware by the author:
the parsers are tested against text in nvidia-smi's documented formats.
"""
from __future__ import annotations

import re

QUERY_FIELDS = "index,name,memory.total,driver_version"
SMI_QUERY_ARGV = ["nvidia-smi", f"--query-gpu={QUERY_FIELDS}", "--format=csv,noheader,nounits"]
SMI_HEADER_ARGV = ["nvidia-smi"]
RAW_LIMIT = 1500

NOTE = ("This shows which GPU the sandbox can see. It does not show that the repository's code ran on it: a script that "
        "never moves anything to CUDA still runs on the CPU.")

_NOT_A_NAME = {"[n/a]", "n/a", "unknown", "[not supported]", "none"}
_CUDA = re.compile(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_gpu_csv(text: str) -> list[dict]:
    """Lines like `0, NVIDIA H100 80GB HBM3, 81559, 550.54.15`. Unparseable lines are skipped, not guessed at."""
    gpus: list[dict] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4 or not (parts[0].isascii() and parts[0].isdigit()) or not parts[1] or parts[1].lower() in _NOT_A_NAME:
            continue
        mem = int(parts[2]) if parts[2].isascii() and parts[2].isdigit() else None          # "[N/A]" on some systems
        gpus.append({"index": int(parts[0]), "name": parts[1][:80], "memory_mib": mem,
                     "driver": parts[3][:40] if re.fullmatch(r"[0-9][0-9.]*", parts[3]) else None})
    return gpus


def parse_cuda_version(text: str) -> str | None:
    m = _CUDA.search(text or "")
    return m.group(1) if m else None


def describe(gpus: list[dict], cuda: str | None) -> str:
    if not gpus:
        return "no GPU"
    names: dict[str, int] = {}
    for g in gpus:
        names[g["name"]] = names.get(g["name"], 0) + 1
    what = ", ".join(f"{n}× {name}" if n > 1 else name for name, n in names.items())
    mems = {g.get("memory_mib") for g in gpus}
    mem = f", {gpus[0]['memory_mib']} MiB each" if len(mems) == 1 and None not in mems else (
        ", memory " + "/".join(f"{g['memory_mib']}" if g.get("memory_mib") else "?" for g in gpus) + " MiB" if len(gpus) > 1 else "")
    drivers = {g.get("driver") for g in gpus}
    drv = gpus[0].get("driver") if len(drivers) == 1 else None
    return what + mem + (f", driver {drv}" if drv else "") + (f", CUDA {cuda} (driver's maximum)" if cuda else "")


def record(*, requested: bool, status: str, probed_in: str | None, gpus: list[dict] | None = None, cuda: str | None = None,
           detail: str = "", raw: str = "") -> dict:
    gpus = gpus or []
    return {"requested": requested, "status": status, "probed_in": probed_in, "gpus": gpus, "cuda_driver": cuda,
            "summary": describe(gpus, cuda) if gpus else {"not_requested": "CPU only (GPU not enabled)",
                                                          "none_visible": "no GPU visible",
                                                          "probe_failed": "GPU probe failed"}.get(status, "unknown"),
            "detail": detail, "raw": raw[:RAW_LIMIT], "note": NOTE}


def from_smi(*, requested: bool, probed_in: str, query_stdout: str, query_code: int | None, query_stderr: str = "",
             header_stdout: str = "") -> dict:
    """Turn nvidia-smi output into the report's hardware record."""
    gpus = parse_gpu_csv(query_stdout) if query_code == 0 else []
    raw = (query_stdout or "").strip()
    if gpus:
        return record(requested=requested, status="detected", probed_in=probed_in, gpus=gpus, cuda=parse_cuda_version(header_stdout),
                      detail="as reported by `nvidia-smi --query-gpu=" + QUERY_FIELDS + "`", raw=raw)
    err = (query_stderr or query_stdout or "").strip().splitlines()
    last = err[-1][:200] if err else f"exit code {query_code}"
    if query_code == 0 and not raw:
        return record(requested=requested, status="none_visible", probed_in=probed_in, detail="nvidia-smi ran but listed no GPU", raw=raw)
    if query_code == 0:                                  # it printed something, but not lines this parser understands
        return record(requested=requested, status="probe_failed", probed_in=probed_in,
                      detail="nvidia-smi printed output that could not be read as a GPU list (shown verbatim below)", raw=raw)
    return record(requested=requested, status="probe_failed", probed_in=probed_in, detail=f"nvidia-smi failed: {last}", raw="\n".join(err)[-RAW_LIMIT:])
