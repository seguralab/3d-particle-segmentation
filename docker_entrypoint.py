#!/usr/bin/env python
"""
Docker entrypoint for 3D particle segmentation.

Thin CLI wrapper that accepts filename and parameter overrides via command-line
arguments, then delegates to the segmentation pipeline.
"""

import argparse
import os
import sys
from segmentation_processing import (
    detect_input_type,
    load_and_process_input,
    get_default_params,
    validate_required_params,
    run_segmentation,
)
from resize_tif import get_pixel_and_z_dimensions


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Particle Segmentation Pipeline",
    )

    # Required
    parser.add_argument(
        "--filename", required=True, help="Input filename (e.g. sample.tif)"
    )

    # Directories
    parser.add_argument(
        "--input-dir", default="./input", help="Input directory (default: ./input)"
    )
    parser.add_argument(
        "--output-dir", default="./output", help="Output directory (default: ./output)"
    )

    # Parameter overrides
    parser.add_argument("--dx", type=float, help="Voxel size X (micrometers)")
    parser.add_argument("--dy", type=float, help="Voxel size Y (micrometers)")
    parser.add_argument("--dz", type=float, help="Voxel size Z (micrometers)")
    parser.add_argument("--dxyz", type=float, help="Resized voxel size (micrometers)")
    parser.add_argument("--th", type=int, help="Intensity threshold")
    parser.add_argument(
        "--radius-um", type=float, help="Expected bead radius (micrometers)"
    )
    parser.add_argument("--s2v-max", type=float, help="Max surface-to-volume ratio")
    parser.add_argument(
        "--fluorescent-label",
        type=int,
        choices=[0, 1],
        help="1=beads labeled, 0=void labeled",
    )
    parser.add_argument(
        "--crop-bool", type=int, choices=[0, 1], help="1=crop, 0=full image"
    )
    parser.add_argument("--channel-num", type=int, help="Channel number (LIF files)")
    parser.add_argument("--example-frame", type=int, help="Z-slice for visualization")
    parser.add_argument("--peak-prom", type=float, help="Peak prominence for seed detection (auto: radius/5)")
    parser.add_argument("--d-peak", type=float, help="Min distance between peaks in voxels (auto: radius)")
    parser.add_argument("--inten-max", type=float, help="Max intensity always foreground (auto: th*3.33)")
    parser.add_argument("--th-relative", type=float, help="Relative brightness threshold (auto: th/3)")

    # Output options
    parser.add_argument(
        "--no-plot", action="store_true", help="Disable visualization plots"
    )
    parser.add_argument(
        "--no-smooth", action="store_true", help="Disable morphological smoothing"
    )
    parser.add_argument("--no-png", action="store_true", help="Disable PNG output")
    parser.add_argument("--no-mat", action="store_true", help="Disable MAT output")
    parser.add_argument("--no-json", action="store_true", help="Disable JSON output")

    return parser.parse_args()


def main():
    args = parse_args()

    filepath = os.path.join(args.input_dir, args.filename)

    # run_segmentation writes to os.path.join('output', stem).
    # If --output-dir differs from the default, symlink ./output to the target
    # so the pipeline writes to the right place without code changes.
    output_dir = os.path.normpath(args.output_dir)
    default_output = os.path.normpath("./output")
    if output_dir != default_output:
        os.makedirs(output_dir, exist_ok=True)
        if os.path.islink(default_output):
            os.remove(default_output)
        if not os.path.exists(default_output):
            os.symlink(os.path.abspath(output_dir), default_output)

    # Detect input type
    input_type = detect_input_type(args.filename)
    print(f"Detected input type: {input_type}")

    # Load defaults then apply CLI overrides
    params = get_default_params(input_type)

    override_map = {
        "dx": args.dx,
        "dy": args.dy,
        "dz": args.dz,
        "dxyz": args.dxyz,
        "th": args.th,
        "radius_um": args.radius_um,
        "s2v_max": args.s2v_max,
        "fluorescent_label": args.fluorescent_label,
        "crop_bool": args.crop_bool,
        "channel_num": args.channel_num,
        "example_frame": args.example_frame,
        "peak_prom": args.peak_prom,
        "d_peak": args.d_peak,
        "inten_max": args.inten_max,
        "th_relative": args.th_relative,
    }
    overrides = {k: v for k, v in override_map.items() if v is not None}
    if overrides:
        params.update(overrides)
        print(f"Applied overrides: {list(overrides.keys())}")

    # For TIF files, attempt to read dx/dy/dz from file metadata before
    # validation so we can include detected values in any error message.
    detected_meta = {}
    if input_type == 'tif':
        try:
            meta_dx, meta_dy, meta_dz = get_pixel_and_z_dimensions(filepath)
            if meta_dx is not None:
                detected_meta['dx'] = meta_dx
            if meta_dy is not None:
                detected_meta['dy'] = meta_dy
            if meta_dz is not None:
                detected_meta['dz'] = meta_dz
        except Exception:
            pass

    # Validate required parameters
    try:
        validate_required_params(params, input_type)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        if detected_meta:
            print(f"\nDetected from TIF metadata: {detected_meta}", file=sys.stderr)
            print(
                "Please confirm these values by passing them explicitly "
                "(e.g. --dx {dx} --dy {dy} --dz {dz}).".format(
                    dx=detected_meta.get('dx', '?'),
                    dy=detected_meta.get('dy', '?'),
                    dz=detected_meta.get('dz', '?'),
                ),
                file=sys.stderr,
            )
        print(
            "\nFor TIF/LIF files, you must provide: "
            "--dx, --dy, --dz, --fluorescent-label, --radius-um",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load image
    print(f"Loading: {filepath}")
    img3d = load_and_process_input(filepath, input_type, params)
    raw_shape = params.get('raw_shape', img3d.shape)
    print(f"Image loaded. Raw: {raw_shape[0]}x{raw_shape[1]} px, {raw_shape[2]} slices")
    total_voxels = img3d.shape[0] * img3d.shape[1] * img3d.shape[2]
    dxyz_source = "auto" if args.dxyz is None else "override"
    print(f"Voxelized shape: {img3d.shape} | dxyz: {params['dxyz']:.4f} µm ({dxyz_source}) | Total voxels: {total_voxels:,}")

    # Output options — Docker produces JSON only by default
    output_options = {
        "does_plot": False,
        "further_smooth": not args.no_smooth,
        "save_png": False,
        "save_mat": False,
        "save_json": True,
    }

    # Run pipeline
    run_segmentation(img3d, args.filename, params, output_options)


if __name__ == "__main__":
    main()
