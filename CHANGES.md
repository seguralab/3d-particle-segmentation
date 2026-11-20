# Changes from Vanilla Gong Segmentation

This document summarizes the key modifications made to the original 3D particle segmentation pipeline ([Gong et al.](https://github.com/seguralab/Gong-3d-particle-segmentation)) for integration with the [lovamap](https://github.com/seguralab/lovamap) ecosystem.

## 1. KD-Tree Seed Finding Optimization

**File:** `utils.py` — `find_seeds()`

**Problem:** The original seed-finding step used O(n²) brute-force pair comparison. For every detected peak, it computed the distance to all other peaks using a vectorized broadcast over the full array:

```python
# Original — O(n²)
for ii in range_n_local:
    dist2 = (xc-x1)**2 + (yc-y1)**2 + (zc-z1)**2
    near = (dist2 < d_peak**2).nonzero()[0]
    for jj in near:
        # ... prominence check
```

For large volumes with tens of thousands of local maxima, this evaluates billions of distance comparisons.

**Fix:** Replaced with `scipy.spatial.cKDTree`, which builds a spatial index in O(n log n) and returns only the pairs within `d_peak`:

```python
# Current — O(n log n + k)
tree = cKDTree(coords)
pairs = tree.query_pairs(d_peak, output_type='ndarray')
```

The prominence-check logic for each pair is unchanged — only the pair-finding step was optimized. For a typical volume with 10,000+ seeds, this reduces seed-finding from minutes to seconds.

## 2. Adaptive `min_distance` for Peak Detection

**File:** `utils.py` — `find_seeds()`, `segmentation_processing.py` — `run_segmentation()`

**Problem:** The original code called `peak_local_max` with the scikit-image default `min_distance=1`, which detects every locally-maximum voxel. For smooth distance transforms of packed particles, this produces a massive number of near-duplicate peaks clustered around each true center — all of which then had to pass through the pair-proximity filter.

**Fix:** Auto-derive `min_distance` from the expected particle radius:

```python
min_distance = max(1, int(d_peak / 4))
```

Where `d_peak ≈ radius_um / dxyz` (particle radius in voxels). This pre-suppresses trivially close peaks before they enter the KD-tree step, drastically reducing the number of raw peaks and close pairs to evaluate.

**Combined effect:** Changes 1 and 2 together provide multiplicative speedup. The `min_distance` heuristic reduces the peak count by 10-100×, and the KD-tree makes the remaining pair evaluations O(n log n) instead of O(n²).

## 3. Intensity Normalization (Percentile Stretch)

**File:** `resize_tif.py` — `convert_tif()`

**Problem:** The original pipeline only performed per-slice floor removal (subtracting the 1st percentile), but never scaled the intensity range. Different samples produce wildly different raw intensity ranges depending on exposure time, laser power, and detector gain. This made the intensity threshold (`--th`) highly sample-dependent — a value of 150 might capture all beads in one sample and miss them entirely in another.

**Fix:** After per-slice floor removal, apply a global percentile stretch so the 99th percentile maps to 255:

```python
# Step 1: Per-slice floor removal (unchanged)
img3d = np.dstack([z - np.percentile(z, 1) for z in img_stack])

# Step 2: Global percentile stretch (new)
img3d = np.clip(img3d, 0, None)
p99 = np.percentile(img3d, 99)
if p99 > 0:
    img3d = img3d / p99 * 255
```

This maps the useful dynamic range of every sample to a consistent 0–255 scale, making threshold values transferable across samples. A threshold of ~133 now consistently means "above ~52% of the useful dynamic range" regardless of acquisition settings.

**Impact on existing parameters:**
- Already-binarized images (P99=255) are unaffected — the stretch is a no-op
- For continuous-intensity images, `--th` values will need to be re-tuned once against the normalized scale, but will then be reusable across samples with similar structure

## 4. Auto-Threshold via Otsu's Method

**File:** `docker_entrypoint.py`

**Problem:** The original pipeline required a manually specified intensity threshold (`--th`), which was highly sample-dependent. The default value of 150 rarely worked across different samples, forcing users into a trial-and-error loop of repeated full segmentation runs.

**Fix:** When `--th` is not provided, the pipeline automatically computes the optimal threshold using Otsu's method (`skimage.filters.threshold_otsu`) on the loaded, normalized 3D volume:

```python
from skimage.filters import threshold_otsu
th = int(round(threshold_otsu(img3d)))
```

Otsu's method finds the threshold that maximally separates foreground from background based on the intensity histogram. Combined with the percentile stretch (Change 3), this produces consistent, automatic binarization across samples without user intervention.

Additionally, a `--binarize` flag converts the entire volume to 0/255 using the threshold (Otsu or `--th`) before segmentation begins. This can be useful for samples where the multi-condition thresholding inside the pipeline produces noisy masks. In practice, for clean fluorescent images, binarization produces identical results to the default pipeline since the internal thresholding already effectively binarizes.

**Result:** Users can run the full pipeline without specifying `--th` at all. The auto-threshold is printed to stdout for transparency. If needed, `--th` can still be passed to override the automatic value.

## 5. Threshold Preview Mode

**Files:** `docker_entrypoint.py`, `segmentation_processing.py` — `generate_threshold_preview()`

**Problem:** Finding the right `--th` value required running the full segmentation pipeline repeatedly (15–30+ seconds per run), with no visual feedback on how threshold changes affect foreground detection.

**Fix:** A new `--threshold-preview` CLI flag that:

1. Loads only 3 representative raw TIF pages (25%, 50%, 75% Z-depth) — no full stack load, no resize
2. Applies per-slice normalization + percentile stretch (matching the full pipeline)
3. Auto-picks 5 candidate threshold values, or uses user-supplied values
4. Generates composite comparison images: raw grayscale + green-overlay binary mask at each threshold, labeled with foreground percentage
5. Updates `preview_info.json` with threshold preview metadata
6. Exits without running segmentation

**Speed:** Completes in 2–5 seconds for any file size (only 3 pages are read).

**Threshold auto-selection:**
- With `--th N`: spreads candidates around the value: `[0.7N, 0.85N, N, 1.15N, 1.3N]`
- Without `--th`: derives from intensity percentiles of nonzero pixels: `[P70, P80, P88, P93, P97]`
- With `--threshold-values "a,b,c"`: uses exact user-specified values

**Intended workflow:**
1. Upload image → run `--preview-only --threshold-preview` (~3s)
2. Frontend displays raw slices + threshold composites
3. User picks the threshold that best captures their particles
4. Full segmentation runs with the chosen `--th` value

This eliminates the trial-and-error loop and gives users visual confirmation before committing to a long pipeline run.

## Other Changes

- **Docker entrypoint** (`docker_entrypoint.py`): CLI wrapper for all parameters, output directory management, TIF metadata auto-detection for `dx`/`dy`/`dz`
- **Preview mode** (`--preview-only`): Fast input validation + raw slice extraction without segmentation
- **Progress reporting** (`progress.py`): HTTP heartbeat reporter for live pipeline status in the lovamap frontend
- **Resolution presets** (`--resolution normal|high`): Auto-calculate `dxyz` targeting ~50M or ~200M voxels
- **Segmented preview slices**: Post-segmentation cross-section PNGs for visual validation
- **`.A` → `.toarray()`**: Compatibility fix for recent scipy sparse matrix API
