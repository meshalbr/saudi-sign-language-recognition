"""Run heavy jobs gently on a laptop: low process priority and a capped number of CPU threads.

Enabled with the environment variable SSL_GENTLE=1 (threads from SSL_THREADS, default 2).
"""

import os


def be_gentle() -> None:
    if os.environ.get("SSL_GENTLE") != "1":
        return
    threads = os.environ.get("SSL_THREADS", "2")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, threads)
    try:
        import torch

        torch.set_num_threads(int(threads))
    except Exception:  # noqa: BLE001 - torch is optional for extraction
        pass
    if os.name == "nt":
        import ctypes

        below_normal = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, below_normal)
    else:
        os.nice(10)
