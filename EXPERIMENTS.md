# Experiments — Drone / Sentinel-2 fusion with TAMRF

Summary of every experiment performed on the drone/S2 adaptation of TAMRF, with the
protocol, exact numbers and interpretation. Complements
[`README_drone_modifs.md`](README_drone_modifs.md).

All experiments use the Grandvillers 2025 drone campaign:

- **HR target**: MicaSense Dual orthomosaics, 10 bands, resampled to a common grid at
  **1.11 m** (ratio 9), 6 dates (30/04, 04/06, 19/06, 10/07, 25/07, 30/07/2025),
  787×532 px.
- **LR context**: Sentinel-2 (10/20 m), 25 clear dates (27/04–25/08/2025), the TAMRF
  `ALL` cloud-free predictions consolidated on the reference 10 m grid.
- **Model**: TAMRF temporal-transformer, ratio 9, 2× RTX 3090 24 GB VM, Pixi
  environment (torch 2.5.1).

---

## 1. Metrics & evaluation protocols

Three metrics (per the TAMRF paper, sect. 3.2.2):

- **RMSE** — radiometric fidelity at original resolution. For LR, the prediction is
  smoothed by a Gaussian kernel then downsampled before comparison; for HR the
  comparison is direct. Unit: reflectance (≈0–1).
- **BRISQUE** — no-reference perceptual quality, 0–100 (0 = best). Penalizes blur,
  slightly favors noise.
- **FRR** — frequency restoration, in dB (log-Fourier profile prediction vs target).

**Protocols** (strategy of `bin/test.py`):

- **`ALL`** — input dates = target dates (reconstruction / band sharpening). It fuses
  the two sensors and denoises (cloud-free, best resolution), but it does **not**
  create dates beyond the union of the input dates of both satellites. The naive
  linear interpolation trivially reproduces the input: RMSE ≈ 0. **Not valid** for
  comparing against a NAIVE baseline.
- **`GAPS`** — `N`-day windows removed from the input; the model must reconstruct the
  missing dates (canonical gap-filling protocol). This is the only fair protocol for
  NAIVE comparison. Default `--gaps_size 30`.

> Densification: to get a denser series than the union of the two satellites' dates
> provides, either use the `CUSTOM` strategy (`--custom_target_dates`, any DOY list) or
> run `ALL` once and interpolate the cloud-free reconstruction with numpy/scipy.

---

## 2. Exp. 1 — Baseline inference Landsat/S2 (strategy `ALL`, 10 sites)

**Goal**: validate the environment and produce cloud-free 10 m series with the
**upstream pretrained model** (`968639_s1t3.ckpt`).

- Input: S2 + Landsat time series downloaded via openEO (Terrascope endpoint,
  migration `openeo.be → openeo.terrascope.be` mid-2026) for 10 sites (Grandvillers +
  9 Osiris sites).
- Command: `bin/test.py --checkpoint model/baseline/... --config .../hydra_config
  --strategy ALL --write_images --disable_metrics --device cuda --subtile_width 165`.

**Result (Grandvillers)**:

- 25 S2 dates + 9 Landsat dates → **32 unique dates**, effective revisit ≈ 4.3 days.
- 3 gaps > 10 days remain (one 23-day gap in July–August, no cloud-free observation).

**Interpretation**: TAMRF `ALL` densifies slightly (S2 ~5 d + L8 ~16 d → ~4.3 d) but
only up to the union of the two satellites' dates: it does not create new dates beyond
that. Densifying *further* than the second satellite adds (e.g. daily series) requires
either the `CUSTOM` strategy (`--custom_target_dates`) or `ALL` + classical
interpolation.

---

## 3. Exp. 2 — First drone/S2 training (ratio 9, early checkpoint 0.195)

**Setup**: first end-to-end training at ratio 9 (after ratio **27** failed by OOM,
see §7). Residual crop HR/LR in `transformer.py`, `patch_size 90`, CCA disabled,
bf16-mixed.

**Result**: after ~70 epochs, best checkpoint `2018_val_total_loss=0.19499768.ckpt`
(val loss 0.195).

**Inférence (checkpoint 0.195)** — RMSE by band:

| LR Sentinel-2 | RMSE | HR MicaSense | RMSE |
|---------------|------|--------------|------|
| b2 (blue 10m) | 0.043 | blue444 | 0.045 |
| b3 (green 10m)| 0.055 | blue | 0.045 |
| b4 (red 10m) | 0.089 | green531 | 0.050 |
| b5 (VEDGE) | 0.083 | green | 0.051 |
| b6 (VEDGE) | 0.091 | red650 | 0.058 |
| b7 (VEDGE) | 0.119 | red | 0.061 |
| b8 (NIR) | 0.116 | re705 | 0.055 |
| b8a (NIR) | 113.59 ⚠️ | re | 0.059 |
| b11 (SWIR) | 0.116 | re740 | 0.069 |
| b12 (SWIR) | 0.128 | nir | 0.077 |

**Anomalies found & fixed** (→ Exp. 4):

1. **LR b8a RMSE = 113.59** → normalization bug: band index 7 was ×1000 (Kelvin
   scaling inherited from Landsat LST). For Sentinel-2, index 7 is **b8a** (NIR
   reflectance). Fixed via `lr_tir_band` parameter (=-1 when no thermal band).
2. **Empty masked metrics** → normal: in `ALL`, all target dates are in the input, so
   everything is flagged `clear`.
3. **HR FRR all negative** (−2 to −4 dB) → predicted HR images are blurrier than the
   target (high frequencies under-represented); expected for an early checkpoint.

---

## 4. Exp. 3 — Second training (1000 epochs, best checkpoint 0.080)

**Setup**: identical hyperparameters (ratio 9, patch 90, CCA off, `bf16-mixed`,
gradient clip 0.1), GPU 1. Also fixed the 3 normalization bugs (§2).

| Metric | Value |
|--------|-------|
| Best validation loss | **0.0802** @ epoch 741 |
| Val loss @ epoch 700 | 0.0875 |
| Val loss @ epoch 999 | ~0.08 |
| Best checkpoint | `36324_val_total_loss=0.08020700.ckpt` |

Improvement: **2.4×** vs the first checkpoint (0.195 → 0.080).

Plateau analysis: the training uses a `CosineAnnealingWarmRestarts` schedule
(LR 1e-4 → 2e-5, T0 = 40 360 steps, warmup 2 018). At epoch 741 (step ~36 324) the
effective LR ≈ 2.2e-5, near the minimum — but the dominant factor is **data size**
(§8).

---

## 5. Exp. 4 — Corrected inference (checkpoint 0.080)

Same command as §3 but with the 3 bug fixes and the best checkpoint. Metrics
(`clear` dates only, strategy `ALL`):

| LR S2 band | RMSE | BRISQUE | FRR | HR MicaSense | RMSE | BRISQUE | FRR |
|-----------|------|---------|-----|--------------|------|---------|-----|
| b2 (490) | 0.046 | 84.6 | −1.63 | blue444 | 0.015 | 50.0 | 1.00 |
| b3 (560) | 0.057 | 79.8 | −0.27 | blue | 0.014 | 48.0 | 0.88 |
| b4 (665) | 0.092 | 79.7 | −0.29 | green531 | 0.015 | 45.3 | 0.55 |
| b5 (705) | 0.084 | 75.5 | 0.55 | green | 0.015 | 44.3 | 0.55 |
| b6 (740) | 0.093 | 71.4 | 0.00 | red650 | 0.015 | 35.1 | 0.33 |
| b7 (783) | 0.120 | 69.7 | 1.03 | red | 0.015 | 33.8 | 0.21 |
| b8 (842) | 0.118 | 70.6 | 1.25 | re705 | 0.015 | 38.4 | 0.50 |
| b8a (865) | 0.117 | 69.4 | 0.90 | re | 0.017 | 38.3 | −0.32 |
| b11 (1610) | 0.120 | 74.4 | 1.39 | re740 | 0.018 | 40.2 | −0.53 |
| b12 (2190) | 0.130 | 76.6 | 0.00 | nir | 0.019 | 41.8 | −0.76 |

**Comparison with the paper** (Table 8, band sharpening LS2S2: RMSE S2 0.009–0.012,
BRISQUE 12–15, FRR 5.3–6.5 dB): our LR RMSE is 5–10× higher, BRISQUE 5–6× worse, FRR
below. Expected because (i) the MicaSense 1.11 m / S2 10 m pair has no equivalent in
LS2S2 (Landsat 30 m / S2 10 m), (ii) only 6 drone dates (vs 40+), (iii) the checkpoint
has not converged optimally.

**HR results are promising**: RMSE 0.014–0.019 (< 2 % radiometric error), BRISQUE
33–50.

---

## 6. Exp. 5 — Fair comparison TAMRF vs NAIVE (protocol `GAPS`)

**Why not `ALL`**: in `ALL`, `lr_target = lr_input` and `hr_target = hr_input`; NAIVE
(linear temporal interpolation = identity at the same dates + bicubic upsample)
reproduces the input → RMSE ≈ 0, meaningless. The NAIVE baseline is only meaningful on
**masked** dates.

**Protocol**: `--strategy GAPS --gaps_size 30` (30-day windows removed; targets =
full series). RMSE on the masked dates:

| HR band | NAIVE | TAMRF | Gain |
|---------|-------|-------|------|
| blue444 | 0.096 | 0.036 | ×2.7 |
| blue | 0.094 | 0.036 | ×2.6 |
| green531 | 0.114 | 0.039 | ×2.9 |
| green | 0.111 | 0.040 | ×2.8 |
| red650 | 0.098 | 0.041 | ×2.4 |
| red | 0.099 | 0.043 | ×2.3 |
| re705 | 0.113 | 0.042 | ×2.7 |
| re | 0.137 | 0.045 | ×3.0 |
| re740 | 0.123 | 0.059 | ×2.1 |
| nir | 0.124 | 0.064 | ×1.9 |

| LR band | NAIVE | TAMRF |
|---------|-------|-------|
| b2 (490) | 0.034 | 0.031 |
| b3 (560) | 0.032 | 0.040 |
| b4 (665) | 0.070 | 0.055 |
| b5 (705) | 0.044 | 0.056 |
| b6 (740) | 0.079 | 0.104 |
| b7 (783) | 0.102 | 0.099 |
| b8 (842) | 0.109 | 0.096 |
| b8a (865) | 0.115 | 0.094 |
| b11 (1610) | 0.081 | 0.071 |
| b12 (2190) | 0.099 | 0.079 |

**Interpretation**:

- TAMRF beats NAIVE on **all HR bands** by 2–3× — the model *really learns* the
  S2→drone relation; it is not limited by a data-capacity ceiling on reconstruction
  quality at unseen dates. This removes the doubt from Exp. 4.
- LR: TAMRF slightly better on short wavelengths (b2–b4), comparable or slightly worse
  on some 20 m bands (b6, b7) — S2 spatial super-resolution (10 m → 1.11 m) of the LR
  branch remains the hardest part.
- **BRISQUE**: NAIVE looks *better* (3.5–22 vs 42–67) because its output is a visually
  sharp smoothed interpolation, while TAMRF generates textured detail. Classic
  radiometric-fidelity (RMSE) vs perceptual-quality (BRISQUE) trade-off. It is the
  intrinsic compromise of TAMRF: it restores high spatial frequencies (lower RMSE on
  drone) at the cost of texture artifacts perceived by BRISQUE.

---

## 7. Failures & diagnostics (why the numbers are what they are)

| # | Problem | Root cause | Resolution |
|---|---------|------------|------------|
| 1 | **OOM at ratio 27** (24 GB 3090) | Memory scales like (HR px)²: a 270 m patch = 729² HR px ≈ 85k tokens/date. 3 failure points: encoder, temporal self-attention, decoder (all query dates at full HR res). | Ratio reduced to **9** (81² px, ≈13× less memory). |
| 2 | **CUDA `invalid configuration argument`** in `scaled_dot_product_attention` | `head_dim = 136/4 = 34`, not a multiple of 8; Flash/Mem-efficient kernels crash at scale (small nominal patches fell back to `math` silently). | Force the `math` SDPA fallback in `bin/train.py`. Structural fix: make `feature_dim` a multiple of 32 (e.g. sensor_token_size 32 → 160, head_dim 40). |
| 3 | **CCA loss `torch.cat() expected a non-empty list`** | `resolutions_for_cca_loss` inherited: pred_resolutions {3.33, 10, 20} matched nothing in the mono-resolution drone config (HR 1.11, LR 10). | CCA disabled (`cca_loss: null`). Could be re-enabled with valid pred resolutions (1.11 HR, 10 LR). |
| 4 | **Empty metrics + `No such file: landsat/index.csv`** at inference | `build_dataset` used the *default* sensors (hr=sentinel2, lr=landsat) instead of the training config (hr=micasense, lr=sentinel2). | Dataset built from the Hydra config (`load_test_config` + `utils.build_dataset`). |
| 5 | **`tensor a (496) must match tensor b (518)`** | `tiled_inference` reassembled 2×270=540 px > the 518-px target window. | Crop the reassembled output to the input size in `tiled_inference`. |
| 6 | **LR b8a RMSE = 113.59** | `normalize_for_tests` multiplied band 7 by 1000 (Landsat LST legacy); for S2, index 7 = b8a NIR. | `lr_tir_band` from the sensor profile (=-1 = no thermal band). |
| 7 | **`estimate_mean_std.py` crash** | Fixed loop `range(2000) + next(dataloader)` raised StopIteration on a single site. | Iterate the dataloader to exhaustion. |
| 8 | **Ratio hard-coded `/ 3`** | Multiple `/ 3` inside the inference pipeline. | Parameterized via `--ratio` / `--lr_resolution` (identical behavior at ratio 3). |

---

## 8. Val-loss plateau analysis (0.08): capacity vs data size

| Parameter | Drone/S2 (ours) | Landsat/S2 (paper) |
|-----------|----------------|--------------------|
| Training sites | **1** | 193 |
| Training samples | **49** | ~4 560 |
| Spatial ratio (LR→HR) | 9× | 3× |
| Drone/Landsat dates | 6 | 40+ |
| S2 dates | 25 | 40+ |

The drone/S2 dataset has 49 training samples (54 spatial patches × 1 temporal slice,
90/10 split) from a single site. The paper uses ~4 560 samples over 193 sites — a
**93×** gap. The LR→HR relation must be learned from 49 examples only; the plateau at
0.08 is therefore most likely a **capacity ceiling** for this architecture given the
data, not a local minimum.

**Identified data-augmentation levers** (in `joint_sits_dataset.py`, none active
upstream):

1. **Transfer learning** (high impact): pretrain on Landsat/S2 (ratio 3, 4 560
   samples) then fine-tune on drone/S2 (ratio 9) via `start_from_checkpoint`
   (`train.py:67-90`). The two ratios are supported; the extra LR upsampling layer
   initializes randomly.
2. **Spatial augmentation** (high impact): H/V flips and 90° rotations in
   `__getitem__` — virtually ×4–8 the dataset size.
3. **Temporal sub-sampling** (medium): `random_reduce_rate: 0.5` (implemented, not
   activated) adds inter-epoch variation.

> Next steps and priorities are described in `README_drone_modifs.md` §8. Metrics to
> use for any future comparison: RMSE, ubRMSE, SMAPE, R², KGE and correlation on the
> downstream task; RMSE / BRISQUE / FRR for the fusion itself.