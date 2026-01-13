"""
3D Particle Segmentation Main Script

Configure the input file and parameters below, then run to segment particles/beads
in 3D microscopy images.
"""

import os
from pathlib import Path
from tqdm import tqdm
from segmentation_processing import detect_input_type, load_and_process_input, get_default_params, run_segmentation


# ============================================================================
#                         CONFIGURATION SECTION
#                   Edit these parameters to customize
# ============================================================================

# Input file configuration
INPUT_DIR = './input/'
FILENAME = 'Rods_3_1_Stack_1A.tif'

# ============================================================================
# PROCESSING PARAMETERS - Override defaults by uncommenting and modifying
# ============================================================================
# 
# Default parameters are automatically loaded based on input file type.
# Uncomment and modify any of the parameters below to customize segmentation:
#

PARAMS_OVERRIDE = {
    # --- Input/Loading Parameters ---
    # 'dx': 1.1375,              # Original voxel size X dimension (micrometers)
    # 'dy': 1.1375,              # Original voxel size Y dimension (micrometers)
    # 'dz': 1.0,                 # Original voxel size Z dimension (micrometers)
    # 'dxyz': 1.5,               # Resized voxel size (micrometers, uniform)
    # 'fluorescent_label': 1,    # 1 = beads are labeled, 0 = void space is labeled
    # 'crop_bool': 0,            # 1 = crop image, 0 = keep full image
    # 'channel_num': 1,          # Channel number (1-4, for LIF files only)
    
    # --- Bead Detection Parameters ---
    # 'radius_um': 50,           # Expected bead radius in micrometers
    # 'peak_prom': None,         # Peak prominence (auto-calculated as radius/5 if not set)
    # 'd_peak': None,            # Distance for merging peaks (auto-calculated as radius if not set)
    
    # --- Intensity Threshold Parameters ---
    # 'th': 150,                 # Absolute intensity threshold for foreground
    # 'inten_max': None,         # Max intensity always considered foreground (auto: th*3.33)
    # 'th_relative': None,       # Relative brightness threshold (auto: th/3)
    
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
    
    # Construct full file path
    filepath = os.path.join(INPUT_DIR, FILENAME)
    
    # Auto-detect input type
    input_type = detect_input_type(FILENAME)
    tqdm.write(f"✓ Detected input type: {input_type}")
    
    # Load default parameters for this file type
    params = get_default_params(input_type)
    tqdm.write(f"✓ Loaded default parameters for {input_type} files")
    
    # Apply any user overrides
    if PARAMS_OVERRIDE:
        params.update(PARAMS_OVERRIDE)
        tqdm.write(f"✓ Applied parameter overrides: {list(PARAMS_OVERRIDE.keys())}")
    
    # Load the image
    tqdm.write(f"✓ Loading image from: {filepath}")
    img3d = load_and_process_input(filepath, input_type, params)
    tqdm.write(f"✓ Image loaded successfully. Shape: {img3d.shape}")
    
    # Run segmentation
    tqdm.write("\n" + "="*60)
    tqdm.write("Starting segmentation pipeline...")
    tqdm.write("="*60 + "\n")
    
    run_segmentation(img3d, FILENAME, params, OUTPUT_OPTIONS)

if __name__ == '__main__':
    main()

