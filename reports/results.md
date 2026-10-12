# Experiment log

| Date | Config | Data | Top-1 | Top-5 | Notes |
|---|---|---|---|---|---|
| 2026-10-12 | karsl100_s01 (12 ep) | KArSL signs 0071-0076 only, signer 01, official split (243 train / 50 test) | 0.88 | 1.00 | smoke run on partially extracted data; validates the loop, not a result |

## Latency (laptop: Ryzen 5 3500U, 4 cores, 7 GB RAM, no CUDA)

| Date | Stage | p50 | p95 | Notes |
|---|---|---|---|---|
| 2026-10-12 | MediaPipe holistic, 640x480 | 71 ms | 84 ms | idle machine |
| 2026-10-12 | BiGRU classifier, PyTorch | 25 ms | 68 ms | measured while 3 extraction workers ran |
| 2026-10-12 | BiGRU classifier, ONNX Runtime CPU | 3.3 ms | 12 ms | same conditions, 2.1 MB model, parity 4.8e-7 |
