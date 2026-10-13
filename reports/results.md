# Experiment log

| Date | Config | Data | Top-1 | Top-5 | Notes |
|---|---|---|---|---|---|
| 2026-10-12 | karsl100_s01 (12 ep) | KArSL signs 0071-0076 only, signer 01, official split (243 train / 50 test) | 0.88 | 1.00 | smoke run on partially extracted data; validates the loop, not a result |
| 2026-10-12 | karsl100_s01 (40 ep) | KArSL-100, signer 01, official split (3606 train / 600 val / 800 test), 100 classes | **0.990** | 0.998 | BiGRU 566k params, best val 0.997 at epoch 32, 37 min on the laptop CPU while 3 extraction workers ran. Signer-dependent: same person in train and test. |
| 2026-10-13 | karsl100_all_official (40 ep) | KArSL-100, signers 01+02+03, official split (10814 train / 1903 val / 2400 test) | **0.996** | 1.000 | signer-dependent, 33 min CPU |
| 2026-10-13 | karsl100_loso_s03 (40 ep) | train s01+s02 (8571 / 1501 val), test all 5045 clips of s03 | 0.683 | 0.891 | signer-independent; val 0.999 vs test 0.683 = signer overfitting |
| 2026-10-13 | karsl100_loso_s02 (40 ep) | train s01+s03 (8554 / 1497 val), test all 5066 clips of s02 | 0.641 | 0.893 | signer-independent |
| 2026-10-13 | karsl100_loso_s01 (40 ep) | train s02+s03 (8610 / 1501 val), test all 5006 clips of s01 | 0.687 | 0.867 | signer-independent |
| 2026-10-13 | leave-one-signer-out mean | 3 folds, 15,117 test clips in total | **0.670** | 0.884 | the number to improve |

## Phase 2: closing the signer gap (fold: test signer 02, baseline 0.641 / 0.893)

| Date | Config | Change vs baseline | Top-1 | Top-5 | Delta top-1 |
|---|---|---|---|---|---|
| 2026-10-13 | p2_s02_handlocal | hand points wrist-relative, scaled by hand size | 0.701 | 0.912 | +6.0 |
| 2026-10-13 | p2_s02_vel | + velocity (frame differences) only | 0.655 | 0.883 | +1.4 |
| 2026-10-13 | p2_s02_handlocal_vel | hand-local + velocity | 0.697 | 0.904 | +5.6 |
| 2026-10-13 | p2_s02_full | hand-local + velocity + hand-block dropout 0.15 + dropout 0.5 | **0.744** | **0.932** | **+10.3** |

Reading: removing each signer's hand size and proportions (hand-local) is the single biggest win. Velocity alone is noise, but it does not hurt when combined. Regularisation (dropout 0.5 and simulated missing hands) adds another 4 to 5 points, which fits the diagnosis of signer memorisation. Known weakness of the protocol: validation top-1 saturates at 1.000 by epoch 15 in every run, so "best checkpoint by validation" is effectively arbitrary; selection by validation loss or a fixed final epoch is the next protocol fix.

Reading of phase 1: with the signer seen in training the model is near perfect; with a new signer it drops to two thirds top-1 while top-5 stays near 0.9. Validation (same signers as train) reaches 1.000 in every fold, so the model memorises signer style. Next experiments target that gap: stronger spatial augmentation, hand-scale normalisation, velocity features, dropout on whole hand blocks, and a Transformer comparison.

## Latency (laptop: Ryzen 5 3500U, 4 cores, 7 GB RAM, no CUDA)

| Date | Stage | p50 | p95 | Notes |
|---|---|---|---|---|
| 2026-10-12 | MediaPipe holistic, 640x480 | 71 ms | 84 ms | idle machine |
| 2026-10-12 | BiGRU classifier, PyTorch | 25 ms | 68 ms | measured while 3 extraction workers ran |
| 2026-10-12 | BiGRU classifier, ONNX Runtime CPU | 3.3 ms | 12 ms | same conditions, 2.1 MB model, parity 4.8e-7 |
