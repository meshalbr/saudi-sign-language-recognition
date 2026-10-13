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

Full recipe on all folds (leave-one-signer-out):

| Date | Fold (test signer) | Baseline top-1 | Full recipe top-1 | Full recipe top-5 |
|---|---|---|---|---|
| 2026-10-13 | s03 | 0.683 | **0.794** | 0.945 |
| 2026-10-13 | s02 | 0.641 | **0.744** | 0.932 |
| 2026-10-13 | s01 | 0.687 | **0.751** | 0.932 |
| 2026-10-13 | mean | 0.670 | **0.763** | 0.936 |

Stronger regularisation ("strong": hand_drop 0.3, weight_decay 1e-3, checkpoint tie-break by validation loss):

| Date | Fold | Full recipe top-1 | Strong top-1 | Strong top-5 |
|---|---|---|---|---|
| 2026-10-13 | s03 | 0.794 | **0.804** | 0.944 |
| 2026-10-13 | s02 | 0.744 | **0.759** | 0.946 |
| 2026-10-13 | s01 | 0.751 | **0.770** | 0.947 |
| 2026-10-13 | mean | 0.763 | **0.778** | **0.946** |

## Model comparison: GRU vs Transformer (same features, same "strong" regularisation)

| Date | Fold | Model | Params | Train time | Top-1 | Top-5 | Test loss |
|---|---|---|---|---|---|---|---|
| 2026-10-13 | s02 | BiGRU 2x128 | 586k | 28 min | 0.759 | 0.946 | 1.04 |
| 2026-10-13 | s02 | Transformer 2 layers, dim 128, 4 heads, CLS token | 450k | 24 min | **0.824** | **0.961** | 0.67 |

Transformer on all folds:

| Date | Fold | GRU strong top-1 | Transformer top-1 | Transformer top-5 | Test loss |
|---|---|---|---|---|---|
| 2026-10-13 | s03 | 0.804 | **0.836** | 0.959 | 0.65 |
| 2026-10-13 | s02 | 0.759 | **0.824** | 0.961 | 0.67 |
| 2026-10-13 | s01 | 0.770 | **0.840** | 0.960 | 0.66 |
| 2026-10-13 | mean | 0.778 | **0.833** | **0.960** | |

The Transformer beats the GRU on every fold, by 5.5 points on average, with fewer parameters (450k vs 586k), faster training (24 vs 28 min) and much lower test loss (0.66 vs about 1.0), which means its confidence is better calibrated. Its best epochs are 38 to 40, so it was still improving at the end of the schedule; a longer schedule is a cheap next experiment. Final signer-independent result for KArSL-100 after one day of work: **0.833 top-1 / 0.960 top-5**, from 0.670 / 0.884 with the first baseline.

## Deployment model

| Date | Config | Data | Top-1 | Top-5 | Notes |
|---|---|---|---|---|---|
| 2026-10-13 | final_tf_all_official | KArSL-100, signers 01+02+03, official split (10814 train / 1903 val / 2400 test) | 0.998 | 1.000 | Transformer, phase-2 features, strong regularisation; 31 min CPU; ONNX 1.98 MB, parity 1.4e-6; the Predictor + ONNX path scores 0.998 on a 400-clip test sample |

## Summary of the signer-gap work

| Step | Mean LOSO top-1 | Gain |
|---|---|---|
| GRU, plain 155-d landmarks | 0.670 | |
| + wrist-local hands, velocity, hand-block dropout, dropout 0.5 | 0.763 | +9.3 |
| + hand_drop 0.3, weight decay 1e-3, val-loss tie-break | 0.778 | +1.5 |
| Transformer encoder instead of GRU | 0.833 | +5.5 |

Reading: removing each signer's hand size and proportions (hand-local) is the single biggest win. Velocity alone is noise, but it does not hurt when combined. Regularisation (dropout 0.5 and simulated missing hands) adds another 4 to 5 points, which fits the diagnosis of signer memorisation. Known weakness of the protocol: validation top-1 saturates at 1.000 by epoch 15 in every run, so "best checkpoint by validation" is effectively arbitrary; selection by validation loss or a fixed final epoch is the next protocol fix.

Reading of phase 1: with the signer seen in training the model is near perfect; with a new signer it drops to two thirds top-1 while top-5 stays near 0.9. Validation (same signers as train) reaches 1.000 in every fold, so the model memorises signer style. Next experiments target that gap: stronger spatial augmentation, hand-scale normalisation, velocity features, dropout on whole hand blocks, and a Transformer comparison.

## Latency (laptop: Ryzen 5 3500U, 4 cores, 7 GB RAM, no CUDA)

| Date | Stage | p50 | p95 | Notes |
|---|---|---|---|---|
| 2026-10-12 | MediaPipe holistic, 640x480 | 71 ms | 84 ms | idle machine |
| 2026-10-12 | BiGRU classifier, PyTorch | 25 ms | 68 ms | measured while 3 extraction workers ran |
| 2026-10-12 | BiGRU classifier, ONNX Runtime CPU | 3.3 ms | 12 ms | same conditions, 2.1 MB model, parity 4.8e-7 |
| 2026-10-13 | Transformer classifier, PyTorch | 3.0 ms | 3.7 ms | one training run in the background |
| 2026-10-13 | Transformer classifier, ONNX Runtime CPU | 1.6 ms | 2.1 ms | 1.98 MB model, parity 1.7e-6 |

## Data quality cleanup (2026-10-13)

MediaPipe lost the hands mid-sign in 22% of KArSL clips (median gap 2 frames); 15 signs had
under 95% hand coverage, worst near the face (pharynx 68%, thermometer 81%). See
`reports/data_quality.md`. Fixes: re-extracted those 15 signs at full resolution with detection
confidence 0.3 (coverage +2 to 3 points each), and `fill_gaps: 8` interpolates hands lost for up
to 8 frames, in training, inference and the reference display.

| Fold | Before (p2 Transformer) | After (p3, cleaned + fill) |
|---|---|---|
| s03 | 0.836 | 0.839 |
| s02 | 0.824 | 0.821 (re-extraction alone: 0.810) |
| s01 | 0.840 | **0.851** |
| mean top-1 / top-5 | 0.833 / 0.960 | **0.837 / 0.959** |

| Held-out accuracy, all 3 folds | Before | After |
|---|---|---|
| the 15 weak signs (2,270 clips) | 82.6% | **85.3%** |
| the other 85 signs (12,847 clips) | 83.5% | 83.4% |

Reading: the overall mean barely moves, but the cleanup does what it targets, +2.7 points on
the badly tracked signs with no cost elsewhere. Deployment model `final_v2` (all signers,
official split): 0.997 top-1.
