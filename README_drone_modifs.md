# TAMRF — Drone / Sentinel-2 adaptation

This document describes the modifications made to the upstream
[`tamrfsits`](https://github.com/Evoland-Land-Monitoring-Evolution/tamrfsits) repository
in order to use TAMRF to **fuse drone multispectral orthomosaics (MicaSense, HR) with
Sentinel-2 time series (LR)**, and to train / run inference for that new sensor pair.

It is self-contained: you do not need any prior knowledge of TAMRF to use it. The
original repository README (`README.org`) still describes the upstream Landsat/Sentinel-2
workflow; this document only covers what changed and how to reproduce the drone/S2
experiments.

All experiments, results and design rationale are also summarized in
[`EXPERIMENTS.md`](EXPERIMENTS.md).

---

## 1. Context

TAMRF (Temporal Attention Multi-Resolution Fusion of Satellite Image Time-Series)
fuses image time series acquired by two sensors with different spatial and temporal
resolution, and outputs **all bands of both sensors, at the best spatial resolution,
and cloud-free**, at any requested date.

The original model was trained on the **Landsat-8/9 (30 m) + Sentinel-2 (10/20 m)**
pair (spatial ratio `3`). The goal of this work was to reuse TAMRF for a much more
demanding pair:

| Role     | Sensor        | Bands | Native resolution       |
|----------|---------------|-------|--------------------------|
| **HR**   | Drone MicaSense Dual | 10   | ≈ 8 cm (resampled to 1.11 m) |
| **LR**   | Sentinel-2    | 10    | 10 m (20 m resampled)    |

Spatial ratio used: **`9`** (i.e. HR at `10/9 ≈ 1.11 m`).

> The drone clusters (6 orthomosaics, Grandvillers 2025) are far fewer than the
> ~4 560 samples used in the paper (193 sites, Landsat/S2). This small dataset size is
> the main limitation of the drone training (see `EXPERIMENTS.md` §8).

---

## 2. What changed since the upstream repository

### 2.1 Code changes (source level)

All changes are **backward compatible**: at ratio `3` (Landsat/S2) they behave
identically to the upstream code.

| File | Change |
|------|--------|
| `bin/test.py` | Added `--ratio` (default `3`) and `--lr_resolution` (default `30.0`) to remove hard-coded `/ 3` resolutions. Infers the sensor profile from the Hydra config instead of hard-coded Landsat/S2 (new `load_test_config`, `get_tamrfsits_metrics_profile`). |
| `bin/train.py` | Forces the **math fallback** of `scaled_dot_product_attention` because TAMRF's `head_dim = 34` is not a multiple of 8 (Flash/Mem-efficient kernels crash at scale). Support `start_from_checkpoint` for transfer learning. |
| `src/tamrfsits/core/esrgan.py` | `nb_layers = round(log_base(r))` instead of `int(factor // base)`. Correct for any power-of-3 ratio (`3, 9, 27, 81, …`). |
| `src/tamrfsits/components/transformer.py` | Spatial **crop** of HR and LR tensors to a common size before the temporal merge (no-op when grids are aligned). |
| `src/tamrfsits/data/joint_sits_dataset.py` | Robust `uint16 → int16` cast, per-date window computation, `min_h/min_w` crop, `generic_downscale`. |
| `src/tamrfsits/validation/utils.py` | `tiled_inference` now takes a `ratio`; crops the reassembled output back to the input size. |
| `src/tamrfsits/validation/strategy.py` | `lr_tir_band` parameterized: the ×1000 (Kelvin) scaling applied to band index 7 only applies when the LR sensor actually has a thermal band. |
| `src/tamrfsits/validation/workflow.py` | `model_predict` propagates `ratio`. Sensor profiles (bands / resolutions / MTF) derived from the config. |
| `bin/sample_ls2s2_fusion.py` | Points to the new Terrascope openEO endpoint (`openeo.terrascope.be/openeo/1.2`). |

### 2.2 New files

- **`bin/convert_drone_to_tamrf.py`** — converts raw drone GeoTIFFs (EPSG:4326, 8 cm,
  uint16) to TAMRF HR format (UTM 31N, `--hr-res`, option `--scale` to map
  reflectance ×65535 → ×10000, alpha band split into a binary mask, `index.csv`).
- **`bin/compute_drone_common_grid.py`** — computes the intersection of the drone
  footprints and produces the common HR grid JSON used by all dates.
- **`bin/consolidate_s2_tamrf.py`** — re-associates the S2 TAMRF `*_pred.tif`
  predictions (issued without georeferencing) with the reference S2 grid, producing
  `tamrf_pairs/sentinel2/`.
- **`bin/auto_download.sh`** — openEO helper that discovers finished jobs via the API
  and generates the `openeo_jobs.json` files (the upstream `run_all.sh` had two bugs:
  it re-submitted in download mode and duplicated the site name in the output path).
- **`bin/estimate_mean_std.py`** — computes `hr_mean/std`, `lr_mean/std` statistics
  for the config (fixed the upstream `StopIteration` crash on a single site).
- **`hydra/datamodule/drone_s2.yaml`**, **`hydra/training_module/drone_fusion.yaml`**,
  **`hydra/location/vm.yaml`** — Hydra configs for the drone/S2 training.
- **`model/drone/hydra_config/config.yaml`** — fully resolved inference config for the
  drone model.

### 2.3 Key design constraints (why these numbers)

These are **structural consequences** of TAMRF's architecture, not bugs:

1. **The spatial ratio must be an exact power of 3** (`3, 9, 27, …`), because each
   `SubPixelConv` layer upscales by ×3 (`esrgan.py`).
2. **`patch_size` must be a multiple of `PPCM(10, r)`** (10 m = LR grid, `r` = ratio).
   For `r = 9`: `PPCM(10, 9) = 90 m` → patch of `9` LR pixels × `81` HR pixels,
   perfectly aligned (no crop needed).
3. **Memory grows like `(HR pixels) × nb dates`**: each HR pixel is an attention token.
   A ratio of **27** (0.37 m) saturated a 24 GB RTX 3090 (OOM); **9** (1.11 m) fits
   comfortably.
4. **The HR sensor carries its own dates** — no temporal coincidence between sensors
   is required (`conjunctions_only: false`). The 6 drone dates are used as HR targets,
   the 25 S2 dates as dense LR context.

---

## 3. Getting the data

### 3.1 Download Sentinel-2 data (optional, public)

To re-download S2 inputs yourself, follow the upstream procedure documented in
[`README.org` → "Download additional data (optional)"](README.org) using
`bin/sample_ls2s2_fusion.py` (submit → download → extract). It requires an
**openEO account on Terrascope** (endpoint `https://openeo.terrascope.be/openeo/1.2`),
the only backend exposing both `SENTINEL2_L2A_SENTINELHUB` and `LANDSAT8-9_L2`.

> **Note: this flow changed slightly from the upstream README.** In June 2026
> Terrascope migrated its openEO endpoint (`openeo.vito.be` → `openeo.terrascope.be/openeo/1.2`);
> the old URL now redirects with HTTP 301, which older openEO clients reject on POST
> (the connection URL in `bin/sample_ls2s2_fusion.py` reflects the new endpoint). The
> `run_all.sh` automation also had two bugs (it re-submitted jobs in `download` mode
> and duplicated the site name in the output path); use `bin/auto_download.sh` instead,
> which discovers finished jobs via the openEO API.

### 3.2 Files to transfer (checklist)

The drone dataset and the trained checkpoints are **not public** and **not in this
repository**. To reproduce the experiments you need the following files:

| # | File / folder | Origin | Used by |
|---|---------------|--------|---------|
| 1 | Raw drone orthomosaics (6 dates, GeoTIFF, 8 cm, 10 bands + alpha) | Nextcloud (Grandvillers 2025) | `bin/convert_drone_to_tamrf.py` |
| 2 | `drone_common_grid_r9.json` | computed by `bin/compute_drone_common_grid.py` | forcing a common grid across drone dates |
| 3 | `tamrf_pairs/micasense/` (6 dates, `index.csv`) | output of conversion | training / inference |
| 4 | `tamrf_pairs/sentinel2/` (25 dates, `index.csv`) | S2 TAMRF `ALL` predictions consolidated by `bin/consolidate_s2_tamrf.py` | training / inference |
| 5 | Drone-trained checkpoint (`36324_val_total_loss=0.08020700.ckpt`) | VM `~/tamrfsits/results/drone_s2/checkpoints/.../default/` | drone/S2 inference |
| 6 | Baseline checkpoint `model/baseline/968639_s1t3.ckpt` + `hydra_config/` | Zenodo `10.5281/zenodo.17474541` (also in this repo) | Landsat/S2 inference |
| 7 | `data/timeseries/<site>/` (S2+Landsat, LS2S2 format) | openEO download (10 sites), **only** required for the baseline Landsat/S2 experiments (Config 1) | Landsat/S2 inference |

> Everything except #5 is either in this repository or reproducible with the scripts
> above. File #5 must be retrieved from the machine where the drone training ran.

---

## 4. Setup

```bash
git clone <your-fork> tamrfsits
cd tamrfsits

# Install the environment (used by upstream, same for drone/S2):
curl -fsSL https://pixi.sh/install.sh | bash
pixi install
source .pixi/envs/default/bin/activate   # or: pixi shell

# Helpful aliases used below
export PY=.pixi/envs/default/bin/python
# Note: the pixi environment contains torch; the conda "soil_moisture_env"
# environment does NOT and is only used for the openEO + ML moisture pipeline.
```

**Place the data** in a `tamrf_pairs/` folder at the repository root
(`tamrf_pairs/micasense/` and `tamrf_pairs/sentinel2/`), or adapt the config:

```bash
# training: point the dataset path to your checkout
python bin/train.py -m datamodule=drone_s2 training_module=drone_fusion \
    hr_encoder=esrenc lr_encoder=esrenc \
    location.drone_s2_dataset_path=/abs/path/to/tamrf_pairs/ \
    location.output_dir=/abs/path/to/results/
```

Two environment variables are recommended (same as upstream):

- `PYTHONOPTIMIZE=TRUE` — disables TAMRF's `assert` (some OpenEO rasters are 1 px off).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — better GPU memory utilization.

---

## 5. Data conversion pipeline (drone raw → TAMRF pairs)

Amortize these commands to prepare the dataset once:

```bash
# 1. Compute the common HR grid (intersection of the 6 drone footprints)
python bin/compute_drone_common_grid.py \
    --input /path/to/drone_orthos \
    --crs EPSG:32631 --hr-res 1.1111 \
    --output drone_common_grid_r9.json

# 2. Convert each date onto that common grid (reflectance x65535 -> x10000)
python bin/convert_drone_to_tamrf.py \
    --input /path/to/drone_orthos \
    --output tamrf_pairs \
    --crs EPSG:32631 --hr-res 1.1111 --scale 0.1526 \
    --grid drone_common_grid_r9.json

# 3. Consolidate the S2 TAMRF predictions (25 dates) as LR pairs
python bin/consolidate_s2_tamrf.py \
    --input /path/to/hr_clr_ALL_50_183/*_pred.tif \
    --output tamrf_pairs \
    --origin-x 467530 --origin-y 5485030 --pixel 10

# 4. (Re-)compute the normalization statistics if you change the data
python bin/estimate_mean_std.py datamodule=drone_s2 training_module=drone_fusion \
    location=vm hr_encoder=esrenc lr_encoder=esrenc
```

**Mask convention (TAMRF semantics — do not swap the polarity).** In TAMRF,
`mask = 1` means *invalid / missing* and `mask = 0` means *valid*
(`core/time_series.py`, docstring of `TimeSeriesData`). When loading, every non-zero
pixel of the `_mask.tif` is masked (`_read_time_series` reads `mask_tif > 0`), and any
reflectance of **0** in the `_bands.tif` is also treated as missing (pixels of value 0
are OR'ed into the mask at `joint_sits_dataset.py`). The conversion scripts therefore
write `0 = valid` on purpose: the drone `_mask.tif` codes `alpha <= 0` as invalid, the
consolidated S2 masks are all-zeros (the dates used are cloud-free). Writing
`255 = valid` (the naive choice) would mask *everything*.

---

## 6. Commands: training & inference

### Config 1 — Baseline Landsat/S2 (public checkpoint, reprepoducible)

Inference with the upstream pretrained model on a site in LS2S2 format (S2+Landsat):

```bash
PYTHONOPTIMIZE=TRUE python bin/test.py \
    --algorithm TAMRFSITS \
    --checkpoint model/baseline/968639_s1t3.ckpt \
    --config model/baseline/hydra_config \
    --ts data/timeseries/Grandvillers \
    --output results/baseline_test \
    --strategy ALL --ratio 3 --lr_resolution 30 \
    --width 990 --subtile_width 165 --margin 30 \
    --device cuda
```

### Config 2 — Drone/S2 training

Train (or fine-tune) the model on the drone/S2 pair (ratio 9, patch 90 m):

```bash
PYTHONOPTIMIZE=TRUE python bin/train.py -m \
    datamodule=drone_s2 training_module=drone_fusion \
    hr_encoder=esrenc lr_encoder=esrenc \
    location.drone_s2_dataset_path=$PWD/tamrf_pairs/ \
    location.output_dir=$PWD/results/
```

- The config (`hydra/datamodule/drone_s2.yaml`) already targets `patch_size: 90`,
  `hr_lr_resolution_ratio: 9`, CCA loss disabled (`cca_loss: null`).
- Best checkpoint with the current data: `results/drone_s2/checkpoints/.../<step>_val_total_loss=0.08020700.ckpt`
  (epoch 741 out of 1000, val loss 0.0802).
- **Transfer learning from Landsat/S2** (recommended next step): pre-train on the
  LS2S2 data then fine-tune with `start_from_checkpoint` (see `train.py:67-90`). The
  LR encoder's extra upsampling layer (ratio 9) is initialized randomly, which is fine.

### Config 3 — Drone/S2 inference

```bash
PYTHONOPTIMIZE=TRUE python bin/test.py \
    --checkpoint results/drone_s2/checkpoints/.../36324_val_total_loss=0.08020700.ckpt \
    --config model/drone/hydra_config \
    --ts tamrf_pairs/ --output data/results/drone/ \
    --strategy ALL --algorithm TAMRFSITS --write_images \
    --device cuda --ratio 9 --lr_resolution 10 \
    --width 576 --subtile_width 270 --margin 5 \
    --show_subtile_progress
```

Notes on the inference flags (specific to the drone case):

- `--width 576` — maximum square window fitting the drone grid (787×532 px @ 1.11 m),
  and a multiple of 9.
- `--subtile_width 270` — HR-pixel subtile side used to keep VRAM under control.
- `--margin 5` — NOT the default 30: the LR grid is only ~57 px, and the metrics
  need `2×margin + 1 < LR size`.
- `--ratio 9 --lr_resolution 10` — required by the codepath since resolutions are no
  longer hard-coded.

### Fair comparison vs NAIVE (gap-filling, not ALL)

The `ALL` strategy copies the input dates to the targets: a linear NAIVE baseline then
trivially reproduces the input (RMSE ≈ 0) and is **not** comparable. Use `GAPS`:

```bash
# TAMRF
python bin/test.py --ts tamrf_pairs/ --output results/drone/tamrf_gaps \
    --checkpoint <drone_ckpt> --config model/drone/hydra_config \
    --algorithm TAMRFSITS --strategy GAPS --gaps_size 30 \
    --ratio 9 --lr_resolution 10 --width 576 --subtile_width 270 --margin 5

# NAIVE
python bin/test.py --ts tamrf_pairs/ --output results/drone/naive_gaps \
    --algorithm NAIVE --strategy GAPS --gaps_size 30
```

On this protocol, TAMRF beats NAIVE by a factor **2–3×** on every HR band
(RMSE 0.036–0.064 vs 0.094–0.137), see `EXPERIMENTS.md` §6.

---

## 7. Re-using the upstream Landsat/S2 model

None of the changes break ratio `3`. The upstream pretrained checkpoint is preserved
locally (`model/baseline/968639_s1t3.ckpt` + `model/baseline/hydra_config/`). The
sensor profile is now derived from the config, and the Landsat thermal band (index 7,
Kelvin ×1000) is handled automatically via `lr_tir_band`. Full command: see
[Config 1](#config-1--baseline-landsats2-public-checkpoint-reprepoducible).

---

## 8. Perspectives

1. **Data augmentation / transfer learning** (highest priority). The val-loss plateau
   at 0.08 is a *capacity* plateau: only 49 training samples (1 site) vs ~4 560 in the
   paper (193 sites). Three levers, in order of expected impact:
   - spatial augmentation (H/V flips, 90° rotations) in `joint_sits_dataset.py` —
     virtually multiplies the dataset by 4–8×;
   - `random_reduce_rate: 0.5` in the datamodule (implemented upstream, not enabled);
   - transfer learning from the Landsat/S2 pretrained model
     (`start_from_checkpoint`).
2. **More drone sites / campaigns** to increase spatial diversity.
3. **Extend the fair baseline comparison** (GAPS protocol) to STAIR, SEN2LIKE, DSTFN,
   DEEPHARMO, DSEN2, now that the NAIVE baseline has been de-hard-coded (bands / ratio
   / TIR / resolutions derived from the sensor profile, like TAMRF).
4. **Integrate the densified S2 series into a downstream task** — e.g. soil-moisture
   prediction at the probe locations: TAMRF `ALL` + interpolation over `tamrf_pairs/`
   produces dense, cloud-free, 18-band time series that replace the sparse raw S2
   inputs of an ML pipeline.
5. **Extension to a thermal band** — the sensor-profile machinery (`lr_tir_band`) is
   ready to return to a Landsat/S2 pair; no code change needed.
6. **Open questions** — LR BRISQUE is high (69–85): investigate whether this is a
   downsampling (MTF) artefact or under-predicted high frequencies. HR FRR values are
   low/negative: confident the model under-reproduces high frequencies? A frequency
   (FFT) loss could help.
7. **Cross-validation on different AOIs** than the training site to measure real
   generalization.

Full breakdown of every experiment, metric tables and interpretation:
[`EXPERIMENTS.md`](EXPERIMENTS.md).