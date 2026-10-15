"""Run heavy jobs gently on a laptop: low process priority and a capped number of CPU threads.

Enabled with SSL_GENTLE=1. Threads from SSL_THREADS (default 2); CPU mask from SSL_CPU_MASK
(hex, default 0x0F = 4 logical CPUs).
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
        from ctypes import wintypes

        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        handle = k.GetCurrentProcess()
        k.SetPriorityClass(handle, 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        # Limit to a subset of logical CPUs (default the first 4 of 8) so the laptop stays cooler
        # no matter how many threads MediaPipe or PyTorch start.
        k.SetProcessAffinityMask(handle, int(os.environ.get("SSL_CPU_MASK", "0x0F"), 16))
    else:
        os.nice(10)
