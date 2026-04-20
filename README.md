# 3D Particle Segmentation

Automated segmentation of 3D particle/bead structures in microscopy images using Python and scikit-image watershed algorithm. 

The output is a JSON configured to work as an input file for [lovamap](https://github.com/seguralab/lovamap). Final analysis data can be found in [lovamap.com](https://lovamap.com).

## Environment Setup

### Required Python Packages

Install dependencies using:
```bash
pip install -r requirements.txt
```

This installs: numpy, scipy, matplotlib, scikit-image, tqdm, and pillow.

### MATLAB Dependencies (Optional)

MATLAB files are provided for preprocessing:
- `resize_confocal.m` - Resizes confocal images and compensates for z-direction intensity nonuniformity
- `resize_SCAPE.m` - Resizes SCAPE images and compensates for x-direction intensity nonuniformity
- These call helper functions: `nd2read3d.m` and `nd2finfo.m`

## Supported Input Formats

- **TIF files** (`.tif`, `.tiff`)
- **Leica files** (`.lif`)
- **MATLAB files** (`.mat` with variable `img3d_resize`)

The input type is detected automatically based on file extension.

## How to Run

There are two ways to use this tool: **standalone** (edit a config file and run directly) or **Docker** (pass everything via command-line arguments). Choose whichever fits your workflow.

### Option A: Standalone Usage

Best for interactive/local work where you tune parameters across multiple runs.

**1. Prepare your input** — place your image file in the `input/` directory.

**2. Edit configuration** — open `segment3d_2.py` and modify the top section:

```python
INPUT_DIR = './input/'
FILENAME = 'your_input_file.tif'

PARAMS_OVERRIDE = {
    'th': 100,
    's2v_max': 0.7,
    'radius_um': 45,
}

OUTPUT_OPTIONS = {
    'does_plot': True,
    'further_smooth': True,
    'save_png': True,
    'save_mat': True,
    'save_json': True,
}
```

**3. Run:**

```bash
python segment3d_2.py
```

Results are saved to `output/{filename}/`.

### Option B: Docker Usage

Best for reproducible runs, CI/CD pipelines, or environments where you don't want to install dependencies locally. All parameters are passed as command-line arguments — no need to edit any files.

**1. Pull or build the image:**

```bash
# Pull from GitHub Container Registry
docker pull ghcr.io/seguralab/3d-particle-segmentation:latest

# Or build locally
docker build -t 3d-particle-segmentation .
```

**2. Run with volume mounts:**

```bash
docker run \
  -v /path/to/your/input:/app/input \
  -v /path/to/your/output:/app/output \
  ghcr.io/seguralab/3d-particle-segmentation:latest \
  --filename your_sample.tif
```

The container reads from `/app/input` and writes to `/app/output`. Mount your host directories to these paths.

**3. Override parameters via CLI flags:**

```bash
docker run \
  -v /data/input:/app/input \
  -v /data/output:/app/output \
  ghcr.io/seguralab/3d-particle-segmentation:latest \
  --filename my_sample.tif \
  --th 120 \
  --radius-um 40 \
  --dxyz 1.5 \
  --s2v-max 0.6
```

**Available CLI flags:**

| Flag | Type | Description |
|------|------|-------------|
| `--filename` | string | **(Required)** Input filename |
| `--input-dir` | string | Input directory (default: `./input`) |
| `--output-dir` | string | Output directory (default: `./output`) |
| `--th` | int | Intensity threshold |
| `--radius-um` | float | Expected bead radius (µm) |
| `--dxyz` | float | Resized voxel size (µm) |
| `--dx`, `--dy`, `--dz` | float | Original voxel dimensions (µm) |
| `--s2v-max` | float | Max surface-to-volume ratio |
| `--fluorescent-label` | 0 or 1 | 1 = beads labeled, 0 = void labeled |
| `--crop-bool` | 0 or 1 | Crop image toggle |
| `--channel-num` | int | Channel number (LIF files) |
| `--example-frame` | int | Z-slice index for visualization |
| `--no-plot` | flag | Disable visualization plots |
| `--no-smooth` | flag | Disable morphological smoothing |
| `--no-png` | flag | Disable PNG output |
| `--no-mat` | flag | Disable MAT output |
| `--no-json` | flag | Disable JSON output |

Any parameter you don't specify uses the default for the detected file type (see [Main Parameters](#main-parameters) below).

**Tip:** You can also run the entrypoint directly without Docker if you have the dependencies installed:

```bash
python docker_entrypoint.py --filename my_sample.tif --th 120
```

## Processing Pipeline

The segmentation workflow follows these steps:

### 1. **Input Detection & Loading** (`detect_input_type()`, `load_and_process_input()`)
   - Automatically detects file type from extension
   - Loads image using appropriate loader (TIF, LIF, or MAT)
   - Applies voxel resizing if needed

### 2. **Binary Thresholding**
   - Applies multiple thresholds to create binary foreground mask
   - Uses absolute intensity threshold (`th`)
   - Uses relative brightness threshold (`th_relative`)
   - Uses maximum intensity threshold (`inten_max`)

### 3. **Distance Transform**
   - Computes Euclidean distance from foreground voxels to background
   - Handles image boundaries to avoid artificial edge detection

### 4. **Seed Detection** (Step 1)
   - Finds local maxima in the distance transform
   - Removes low-prominence peaks using `peak_prom`
   - Merges closely-spaced peaks using `d_peak`

### 5. **Initial Watershed**
   - Performs watershed segmentation from initial seeds
   - Produces initial bead labels

### 6. **Seed Refinement** (Step 2)
   - Removes seeds that produce regions with poor morphology
   - Uses surface-to-volume ratio threshold (`s2v_max`)

### 7. **Second Watershed**
   - Re-performs watershed with refined seeds
   - Produces cleaner initial segmentation

### 8. **Further Segmentation**
   - Attempts to split over-merged regions
   - Uses secondary watershed on regions with multiple local maxima
   - Sorts beads by size

### 9. **Smoothing** (Optional)
   - Removes sharp protruding voxels from bead surfaces
   - Improves morphological quality if `further_smooth = True`

### 10. **Output Generation**
   - **JSON**: Bead count, coordinates, and metadata
   - **MAT files**: Labeled volume before/after smoothing
   - **PNG plots**: Intermediate visualization at each stage

## Main Parameters

Parameters are loaded based on input type using `get_default_params()`. Override defaults in the configuration section.

### Input-Specific Parameters

| Parameter | TIF Default | LIF Default | MAT Default | Description |
|-----------|-------------|-------------|-------------|-------------|
| `dx`, `dy`, `dz` | 1.1375, 1.1375, 1.0 | 0.66, 0.66, 0.2 | N/A | Original voxel dimensions (µm) |
| `dxyz` | 1.5 | 0.8 | 2.0 | Resized voxel dimension (µm) |
| `fluorescent_label` | 1 | 1 | N/A | 1 = beads labeled, 0 = void space labeled |
| `crop_bool` | 0 | False | N/A | Whether to crop image |
| `channel_num` | N/A | 1 | N/A | Channel number for LIF files (1-4) |

### Segmentation Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `radius_um` | 50 | > 0 | Expected bead radius in micrometers - does not need to be accurate |
| `th` | 150 | > 0 | Absolute intensity threshold - voxels with brightness higher than "th" will be considered foreground UNLESS their relative intensity is below "-th_relative" |
| `inten_max` | th × 3.33 | > th | Maximum intensity - voxels with brightness > "inten_max" will always be considered foreground |
| `th_relative` | th / 3 | > 0 | Relative brightness threshold - voxels with brightness below "-th_relative" will always be considered background |
| `peak_prom` | radius / 5 | > 0 | Minimum prominence for seed peaks - for two close local maxima, if the lower local maximum does not have a prominence higher than "peak_prom", it will be removed from the seeds |
| `d_peak` | radius | > 0 | Maximum distance for merging nearby peaks |
| `s2v_max` | 0.65 | > 0 | Maximum surface-to-volume ratio for segmented beads |
| `example_frame` | 20 | < Lz | Z-slice index for visualization |

### Output Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `does_plot` | True / False | Generate intermediate visualization plots |
| `further_smooth` | True / False | Apply additional morphological smoothing |
| `save_png` | True / False | Save visualization PNG files |
| `save_mat` | True / False | Save segmented volume as MAT files |
| `save_json` | True / False | Save metadata as JSON file |

## Troubleshooting Tips

### Segmentation Quality Issues

- **Very bright beads split incorrectly**: Decrease `inten_max`
- **Dark beads ignored**: Increase `th` (intensity threshold)
- **Dark gaps counted as beads**: Decrease `th` or `th_relative`
- **Beads incorrectly split**: 
  - Non-uniform brightness with low minima → Increase `th_relative`
  - Uniform brightness → Increase `peak_prom` or `d_peak`
- **Beads incorrectly merged**: Decrease `peak_prom` or `d_peak`
- **Irregular bead shapes**: Decrease `s2v_max`
- **Small beads not detected**: Increase `s2v_max`
- **Poor segmentation at image edges**: Consider ignoring edge regions if necessary

### General Advice

- Perfect segmentation of all beads is rarely achievable
- Manual merging of components may be necessary
- For problematic regions, consider segmenting a smaller volume around that bead
- Test parameters on a small subset before processing large volumes

## Output Files

For each input file, the script creates a folder `output/{filename}/` containing:

- `{filename}_segment_{th}.mat` - Labeled volume (before smoothing)
- `Smoothed_{filename}_segment_{th}.mat` - Labeled volume (after smoothing)
- `{filename}_segment_{th}.json` - Segmentation metadata and bead data
- Various `*_frame_*.png` files - Visualization plots at different processing stages

## Function Reference

All processing functions and utilities are in `segmentation_processing.py`:

### `detect_input_type(filename)`
Detects the input file type based on file extension.
- Returns: 'tif', 'lif', or 'mat'

### `load_and_process_input(filepath, input_type, params)`
Loads and preprocesses the input image based on its type.
- Returns: 3D numpy array (img3d)

### `get_default_params(input_type)`
Provides sensible default parameters for each input type.
- Returns: Dictionary of parameters

### `run_segmentation(img3d, filename, params, output_options)`
Executes the complete segmentation pipeline.
- Performs all 10 steps of processing
- Saves outputs to `output/{filename}/`

Lower-level segmentation utilities are in utils.py

## Project Structure

```
├── segment3d_2.py              # Standalone entry point - edit config at top of file
├── docker_entrypoint.py        # Docker/CLI entry point - accepts args via command line
├── segmentation_processing.py  # All processing functions and utilities
├── utils.py                    # Lower-level segmentation utilities
├── resize_tif.py              # TIF image loading and resizing
├── resize_confocal.m           # MATLAB preprocessing for confocal images
├── resize_SCAPE.m             # MATLAB preprocessing for SCAPE images
├── Dockerfile                  # Container build definition
├── requirements.txt            # Python dependencies
├── input/                      # Place your input images here
└── output/                     # Segmentation results saved here
    └── {filename}/            # Results organized by input file name
```
