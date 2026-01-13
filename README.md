# 3D Particle Segmentation

Automated segmentation of 3D particle/bead structures in microscopy images using Python and scikit-image watershed algorithm.

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

## How to Run

### 1. Prepare Your Input

Place your input image file in the `input/` directory. Supported formats:
- **TIF files** (`.tif`, `.tiff`)
- **Leica files** (`.lif`)
- **MATLAB files** (`.mat` with variable `img3d_resize`)

### 2. Edit Configuration

Open `segment3d_2.py` and configure the top section:

```python
# Input file configuration
INPUT_DIR = './input/'
FILENAME = 'your_input_file.tif'  # Change this to your file

# Processing parameters - will be auto-loaded based on file type
# You can override defaults here by uncommenting and changing values:
PARAMS_OVERRIDE = {
    'th': 100,          # Change intensity threshold
    's2v_max': 0.7,     # Change surface-to-volume ratio threshold
    'radius_um': 45,    # Change expected bead radius
}

# Output options
OUTPUT_OPTIONS = {
    'does_plot': True,       # Generate intermediate plots
    'further_smooth': True,  # Apply morphological smoothing
    'save_png': True,        # Save plot images
    'save_mat': True,        # Save segmented volumes as .mat files
    'save_json': True,       # Save metadata as .json file
}
```

The input type is detected automatically based on file extension.

### 3. Run the Script

```bash
python segment3d_2.py
```

All outputs will be saved to `output/{filename}/` automatically.

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

All utility functions are in `segmentation_processing.py`:

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

## Project Structure

```
├── segment3d_2.py              # Main entry point - Edit this to configure
├── segmentation_processing.py  # All processing functions and utilities
├── utils.py                    # Lower-level segmentation utilities
├── resize_tif.py              # TIF image loading and resizing
├── resize_confocal.m           # MATLAB preprocessing for confocal images
├── resize_SCAPE.m             # MATLAB preprocessing for SCAPE images
├── requirements.txt            # Python dependencies
├── input/                      # Place your input images here
└── output/                     # Segmentation results saved here
    └── {filename}/            # Results organized by input file name
```

## Example Data

Original example images are available at:
- **Confocal**: [iCloud link](https://www.icloud.com/attachment/?u=...)
- **SCAPE**: `volumedata.mat`
- **Results**: See `results/` folder for examples 