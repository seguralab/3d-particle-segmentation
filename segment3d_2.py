"""
3D Particle Segmentation Main Script

Configure the input file and parameters below, then run to segment particles/beads
in 3D microscopy images.
"""

import os
import time
from pathlib import Path
from tqdm import tqdm
from segmentation_processing import detect_input_type, load_and_process_input, get_default_params, validate_required_params, run_segmentation


# ============================================================================
#                         CONFIGURATION SECTION
#                   Edit these parameters to customize
# ============================================================================

# Input file configuration
INPUT_DIR = './input/'
FILENAME = 'MASK_10p30x30_P12_S2_10xZoom_0.5AD_1ss_93sl-1.tif'
# FILENAME = 'EK081920_FLIP_488void_647gel_20X.tif'

# ============================================================================
# REQUIRED PARAMETERS — These are sample-specific and must be set correctly.
# Wrong values will produce incorrect segmentation results.
# ============================================================================

# PARAMS_REQUIRED = {
#     'dx': 1.5197,              # Pixel width (µm)
#     'dy': 1.5197,              # Pixel height (µm)
#     'dz': 2.0,                 # Slice spacing (µm)
#     'fluorescent_label': 1,    # 1 = beads labeled, 0 = void labeled
#     'radius_um': 50,           # Expected bead radius (µm)
# }

PARAMS_REQUIRED = {
    'dx': 0.1554,            # Pixel width (µm)
    'dy': 0.1554,            # Pixel height (µm)
    'dz': 1.0,                # Slice spacing (µm)
    'fluorescent_label': 1,    # use 1 if particles/scaffold phase is bright
    'radius_um': 50,           # adjust if expected particle radius differs
}

# ============================================================================
# OPTIONAL PARAMETERS — Uncomment and modify to customize segmentation.
# Defaults are loaded automatically based on input file type.
# ============================================================================

PARAMS_OPTIONAL = {
    # --- Input/Loading Parameters ---
    # 'dxyz': 1.5,               # Resized voxel size (µm, uniform) — auto-calculated if not set
    # 'crop_bool': 0,            # 1 = crop image, 0 = keep full image
    # 'channel_num': 1,          # Channel number (1-4, for LIF files only)

    # --- Intensity Threshold Parameters ---
    # 'th': 150,                 # Absolute intensity threshold for foreground
    # 'inten_max': 500,          # Max intensity always considered foreground (default: th * 3.33)
    # 'th_relative': 50,         # Relative brightness threshold (default: th / 3)

    # --- Bead Detection Parameters ---
    # 'peak_prom': 10,           # Peak prominence for seed detection (default: radius / 5)
    # 'd_peak': 50,              # Min distance between peaks in voxels (default: radius)

    # --- Segmentation Quality Parameters ---
    # 's2v_max': 0.65,           # Maximum surface-to-volume ratio of beads

    # --- Visualization Parameters ---
    # 'example_frame': 20,       # Z-slice index for visualization plots (0-based)
}

# ============================================================================
# OUTPUT OPTIONS
# ============================================================================

OUTPUT_OPTIONS = {
    'does_plot': True,       # Generate intermediate visualization plots
    'further_smooth': True,  # Apply morphological smoothing to segment boundaries
    'save_png': True,        # Save visualization plots as PNG files
    'save_mat': True,        # Save segmented volumes as MATLAB .mat files
    'save_json': True,       # Save segmentation metadata as JSON file
}

# ============================================================================

def main():
    """Main segmentation pipeline"""
    workflow_start = time.perf_counter()
    
    try:
        # Construct full file path
        filepath = os.path.join(INPUT_DIR, FILENAME)
        
        # Auto-detect input type
        input_type = detect_input_type(FILENAME)
        tqdm.write(f"✓ Detected input type: {input_type}")
        
        # Load default parameters for this file type
        params = get_default_params(input_type)
        tqdm.write(f"✓ Loaded default parameters for {input_type} files")

        # Apply required and optional overrides
        params.update(PARAMS_REQUIRED)
        params.update(PARAMS_OPTIONAL)
        applied = list(PARAMS_REQUIRED.keys()) + list(PARAMS_OPTIONAL.keys())
        tqdm.write(f"✓ Applied parameter overrides: {applied}")

        # Validate required parameters
        validate_required_params(params, input_type)

        # Load the image
        tqdm.write(f"✓ Loading image from: {filepath}")
        img3d = load_and_process_input(filepath, input_type, params)
        raw_shape = params.get('raw_shape', img3d.shape)
        tqdm.write(f"✓ Image loaded successfully. Raw: {raw_shape[0]}x{raw_shape[1]} px, {raw_shape[2]} slices")
        total_voxels = img3d.shape[0] * img3d.shape[1] * img3d.shape[2]
        dxyz_source = "auto" if 'dxyz' not in PARAMS_OPTIONAL else "override"
        tqdm.write(f"✓ Voxelized shape: {img3d.shape} | dxyz: {params['dxyz']:.4f} µm ({dxyz_source}) | Total voxels: {total_voxels:,}")

        # Run segmentation
        tqdm.write("\n" + "="*60)
        tqdm.write("Starting segmentation pipeline...")
        tqdm.write("="*60 + "\n")
        
        run_segmentation(img3d, FILENAME, params, OUTPUT_OPTIONS)
    finally:
        tqdm.write(f"Total workflow runtime: {time.perf_counter() - workflow_start:.2f} s")

if __name__ == '__main__':
    main()

