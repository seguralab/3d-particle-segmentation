#!/usr/bin/env python
"""
Docker entrypoint for 3D particle segmentation.

Thin CLI wrapper that accepts filename and parameter overrides via command-line
arguments, then delegates to the segmentation pipeline.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile

import numpy as np
from skimage.filters import threshold_otsu

from segmentation_processing import (
    compute_auto_dxyz,
    detect_input_type,
    generate_threshold_preview,
    load_and_process_input,
    get_default_params,
    validate_required_params,
    run_segmentation,
    RESOLUTION_PRESETS,
)
from resize_tif import get_pixel_and_z_dimensions
from progress import ProgressReporter


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
        "--radius-um", "--radius_um", type=float, help="Expected bead radius (micrometers)"
    )
    parser.add_argument("--s2v-max", "--s2v_max", type=float, help="Max surface-to-volume ratio")
    parser.add_argument(
        "--fluorescent-label", "--fluorescent_label",
        type=int,
        choices=[0, 1],
        help="1=beads labeled, 0=void labeled",
    )
    parser.add_argument(
        "--crop-bool", "--crop_bool", type=int, choices=[0, 1], help="1=crop, 0=full image"
    )
    parser.add_argument("--channel-num", "--channel_num", type=int, help="Channel number (LIF files)")
    parser.add_argument("--example-frame", "--example_frame", type=int, help="Z-slice for visualization")
    parser.add_argument("--peak-prom", "--peak_prom", type=float, help="Peak prominence for seed detection (auto: radius/5)")
    parser.add_argument("--d-peak", "--d_peak", type=float, help="Min distance between peaks in voxels (auto: radius)")
    parser.add_argument("--inten-max", "--inten_max", type=float, help="Max intensity always foreground (auto: th*3.33)")
    parser.add_argument("--th-relative", "--th_relative", type=float, help="Relative brightness threshold (auto: th/3)")

    # Resolution preset
    parser.add_argument(
        "--resolution", choices=["normal", "high"], default="normal",
        help="Resolution preset: normal (~50M voxels) or high (~200M voxels). Default: normal. Ignored if --dxyz is set.",
    )

    # Pre-binarize
    parser.add_argument(
        "--binarize", action="store_true",
        help="Binarize the image using Otsu's method (or --th) before segmentation. "
             "Converts all pixels to 0 or 255.",
    )

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

    # Preview mode
    parser.add_argument(
        "--preview-only", action="store_true",
        help="Load, validate, and save preview slices without running segmentation",
    )

    # Threshold preview mode
    parser.add_argument(
        "--threshold-preview", action="store_true",
        help="Generate threshold comparison images for representative slices and exit without segmentation",
    )
    parser.add_argument(
        "--threshold-values",
        type=str,
        default=None,
        help="Comma-separated threshold values for threshold preview (e.g. '100,130,150,170,200')",
    )

    # Progress / heartbeat reporting
    parser.add_argument(
        "--progress-url", default=None,
        help="Heartbeat endpoint URL (e.g. http://wrapperapi:8080/heartbeat)",
    )
    parser.add_argument(
        "--progress-header", default=None,
        help="Auth header as name:value (e.g. X-Heartbeat-Token:abc123)",
    )
    parser.add_argument(
        "--job-id", default=None,
        help="Job identifier sent with heartbeat messages",
    )

    return parser.parse_args()


def main():
    workflow_start = time.perf_counter()
    args = parse_args()

    try:
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

        # For TIF files, auto-detect dx/dy/dz from metadata as fallback
        # for any values not explicitly provided via CLI.
        metadata_detected = {}
        if input_type == 'tif':
            try:
                meta_dx, meta_dy, meta_dz = get_pixel_and_z_dimensions(filepath)
                for key, val in [('dx', meta_dx), ('dy', meta_dy), ('dz', meta_dz)]:
                    if val is not None and key not in overrides:
                        params[key] = val
                        metadata_detected[key] = val
                if metadata_detected:
                    print(f"Auto-detected from TIF metadata: {metadata_detected}")
            except Exception as meta_err:
                print(f"Metadata detection failed for {filepath}: {meta_err}", file=sys.stderr)

        # Preview-only mode: read raw slices directly (no resize) and exit.
        # Only needs dx/dy/dz for the metadata JSON — skip full validation
        # (fluorescent_label, radius_um are segmentation params, not needed here).
        if args.preview_only:
            name = Path(args.filename).stem
            preview_dir = os.path.join('output', name, 'previews')
            os.makedirs(preview_dir, exist_ok=True)

            if input_type == 'tif':
                with tifffile.TiffFile(filepath) as tif:
                    Lz = len(tif.pages)
                    page_shape = tif.pages[0].shape
                    if len(page_shape) != 2:
                        print(
                            f"Error: TIF has {len(page_shape)+1}D pages (shape={page_shape}). "
                            "Multi-channel TIF files are not supported.",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                    Lx, Ly = page_shape

                    slice_specs = [
                        ("input_slice_25.png", Lz // 4),
                        ("input_slice_50.png", Lz // 2),
                        ("input_slice_75.png", 3 * Lz // 4),
                    ]
                    for slice_name, z_idx in slice_specs:
                        page_data = tif.pages[z_idx].asarray()
                        plt.imsave(
                            os.path.join(preview_dir, slice_name),
                            page_data,
                            cmap='gray',
                        )
            else:
                # Non-TIF: fall back to full load for slice extraction
                img3d = load_and_process_input(filepath, input_type, params)
                Lx, Ly, Lz = img3d.shape
                slice_specs = [
                    ("input_slice_25.png", Lz // 4),
                    ("input_slice_50.png", Lz // 2),
                    ("input_slice_75.png", 3 * Lz // 4),
                ]
                for slice_name, z_idx in slice_specs:
                    plt.imsave(
                        os.path.join(preview_dir, slice_name),
                        img3d[:, :, z_idx],
                        cmap='gray',
                    )

            voxel_size_sources = {}
            for key in ('dx', 'dy', 'dz'):
                if key in overrides:
                    voxel_size_sources[key] = "cli"
                elif key in metadata_detected:
                    voxel_size_sources[key] = "tif_metadata"
                elif key in params:
                    voxel_size_sources[key] = "default"

            dx_val = params.get('dx')
            dy_val = params.get('dy')
            dz_val = params.get('dz')
            if args.dxyz is not None:
                preview_dxyz = args.dxyz
            elif dx_val and dy_val and dz_val:
                target_voxels = RESOLUTION_PRESETS[args.resolution]
                preview_dxyz = compute_auto_dxyz(Lx, Ly, Lz, dx_val, dy_val, dz_val, target_voxels)
            else:
                preview_dxyz = None

            if preview_dxyz and dx_val and dy_val and dz_val:
                est_Lx = round(Lx * dx_val / preview_dxyz)
                est_Ly = round(Ly * dy_val / preview_dxyz)
                est_Lz = round(Lz * dz_val / preview_dxyz)
                est_domain_voxels = est_Lx * est_Ly * est_Lz
            else:
                est_domain_voxels = None

            preview_info = {
                "filename": args.filename,
                "dimensions": {"x": int(Lx), "y": int(Ly), "z": int(Lz)},
                "voxel_size": {
                    "dx": dx_val,
                    "dy": dy_val,
                    "dz": dz_val,
                },
                "voxel_size_sources": voxel_size_sources,
                "resized_voxel_size": preview_dxyz,
                "estimated_domain_voxels": est_domain_voxels,
                "slices": [
                    {"name": sname, "z_index": z_idx}
                    for sname, z_idx in slice_specs
                ],
            }

            with open(os.path.join(preview_dir, 'preview_info.json'), 'w') as f:
                json.dump(preview_info, f, indent=2)

            print(f"Preview saved to {preview_dir}")
            if not args.threshold_preview:
                sys.exit(0)

        # Threshold preview mode: generate composite comparison images and exit.
        if args.threshold_preview:
            name = Path(args.filename).stem
            preview_dir = os.path.join('output', name, 'previews')
            os.makedirs(preview_dir, exist_ok=True)

            fluorescent_label = params.get('fluorescent_label', 1)

            # Load 3 representative raw pages and normalize
            slice_specs = []
            normalized_slices = []

            if input_type == 'tif':
                with tifffile.TiffFile(filepath) as tif:
                    Lz = len(tif.pages)
                    page_shape = tif.pages[0].shape
                    if len(page_shape) != 2:
                        print(
                            f"Error: TIF has {len(page_shape)+1}D pages (shape={page_shape}). "
                            "Multi-channel TIF files are not supported.",
                            file=sys.stderr,
                        )
                        sys.exit(1)

                    for label_str, z_idx in [("25", Lz // 4), ("50", Lz // 2), ("75", 3 * Lz // 4)]:
                        page_data = tif.pages[z_idx].asarray().astype(np.float64)
                        # Normalize per-slice matching pipeline logic (resize_tif.py:77-80)
                        if fluorescent_label == 1:
                            normed = page_data - np.percentile(page_data, 1)
                        else:
                            normed = (page_data - np.percentile(page_data, 1)) * -1 + 255
                        normed = np.clip(normed, 0, None)
                        slice_specs.append((label_str, z_idx))
                        normalized_slices.append(normed)
            else:
                # Non-TIF: fall back to full load
                img3d = load_and_process_input(filepath, input_type, params)
                Lz = img3d.shape[2]
                for label_str, z_idx in [("25", Lz // 4), ("50", Lz // 2), ("75", 3 * Lz // 4)]:
                    normed = img3d[:, :, z_idx].astype(np.float64)
                    normed = np.clip(normed, 0, None)
                    slice_specs.append((label_str, z_idx))
                    normalized_slices.append(normed)

            # Percentile stretch: scale so P99 across all slices maps to 255.
            # This makes threshold values consistent regardless of sample brightness.
            all_pixels = np.concatenate([s.ravel() for s in normalized_slices])
            p99 = np.percentile(all_pixels, 99)
            if p99 > 0:
                normalized_slices = [np.clip(s / p99 * 255, 0, 255) for s in normalized_slices]
                print(f"Percentile stretch: P99={p99:.1f} → mapped to 255")

            # Determine candidate thresholds
            if args.threshold_values:
                threshold_values = [int(v.strip()) for v in args.threshold_values.split(',')]
                threshold_source = "user_specified"
            elif args.th is not None:
                th = args.th
                threshold_values = [
                    int(round(0.7 * th)),
                    int(round(0.85 * th)),
                    th,
                    int(round(1.15 * th)),
                    int(round(1.3 * th)),
                ]
                threshold_source = "auto_from_th"
            else:
                # Auto from intensity percentiles of nonzero pixels across all slices
                all_nonzero = np.concatenate([s[s > 0].ravel() for s in normalized_slices])
                if len(all_nonzero) > 0:
                    threshold_values = [
                        int(round(np.percentile(all_nonzero, p))) for p in [70, 80, 88, 93, 97]
                    ]
                else:
                    threshold_values = [50, 100, 150, 200, 250]
                threshold_source = "auto_from_percentiles"

            # Remove duplicate thresholds while preserving order
            seen = set()
            unique_thresholds = []
            for v in threshold_values:
                if v not in seen:
                    seen.add(v)
                    unique_thresholds.append(v)
            threshold_values = unique_thresholds

            print(f"Threshold preview: {threshold_values} (source: {threshold_source})")

            # Generate composite images for each slice
            slice_metadata = []
            for (label_str, z_idx), normed in zip(slice_specs, normalized_slices):
                meta = generate_threshold_preview(normed, threshold_values, label_str, preview_dir)
                meta["z_index"] = int(z_idx)
                meta["foreground_pcts"] = [t["foreground_pct"] for t in meta["thresholds"]]
                slice_metadata.append(meta)

            # Update preview_info.json
            preview_info_path = os.path.join(preview_dir, 'preview_info.json')
            if os.path.exists(preview_info_path):
                with open(preview_info_path, 'r') as f:
                    preview_info = json.load(f)
            else:
                preview_info = {"filename": args.filename}

            preview_info["threshold_preview"] = {
                "thresholds": threshold_values,
                "threshold_source": threshold_source,
                "slices": [
                    {
                        "name": m["name"],
                        "z_index": m["z_index"],
                        "foreground_pcts": m["foreground_pcts"],
                    }
                    for m in slice_metadata
                ],
            }

            with open(preview_info_path, 'w') as f:
                json.dump(preview_info, f, indent=2)

            print(f"Threshold preview saved to {preview_dir}")
            sys.exit(0)

        # Validate required parameters (full segmentation needs all of them)
        try:
            validate_required_params(params, input_type)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            print(
                "\nFor TIF/LIF files, you must provide: "
                "--dx, --dy, --dz, --fluorescent-label, --radius-um",
                file=sys.stderr,
            )
            sys.exit(1)

        # Apply resolution preset (only matters when dxyz is auto-calculated)
        params['target_voxels'] = RESOLUTION_PRESETS[args.resolution]
        print(f"Resolution: {args.resolution} (~{params['target_voxels']:.0e} target voxels)")

        # Load image
        print(f"Loading: {filepath}")
        img3d = load_and_process_input(filepath, input_type, params)
        raw_shape = params.get('raw_shape', img3d.shape)
        print(f"Image loaded. Raw: {raw_shape[0]}x{raw_shape[1]} px, {raw_shape[2]} slices")
        total_voxels = img3d.shape[0] * img3d.shape[1] * img3d.shape[2]
        dxyz_source = "auto" if args.dxyz is None else "override"
        print(f"Voxelized shape: {img3d.shape} | dxyz: {params['dxyz']:.4f} µm ({dxyz_source}) | Total voxels: {total_voxels:,}")

        # Auto-threshold via Otsu if --th was not explicitly provided
        if args.th is None:
            otsu_th = int(round(threshold_otsu(img3d)))
            params['th'] = otsu_th
            print(f"Auto-threshold (Otsu): {otsu_th}")
        else:
            print(f"Using user threshold: {params['th']}")

        # Pre-binarize: convert image to 0/255 using the threshold
        if args.binarize:
            th_val = params['th']
            img3d = np.where(img3d > th_val, 255.0, 0.0)
            print(f"Binarized image at th={th_val}")

        # Output options - Docker produces JSON only by default
        output_options = {
            "does_plot": False,
            "further_smooth": not args.no_smooth,
            "save_png": False,
            "save_mat": False,
            "save_json": True,
        }

        # Build progress reporter
        auth_header = None
        if args.progress_header:
            name_val = args.progress_header.split(":", 1)
            if len(name_val) == 2:
                auth_header = (name_val[0], name_val[1])
        if auth_header is None:
            token = os.environ.get("HEARTBEAT_TOKEN")
            if token:
                auth_header = ("X-Heartbeat-Token", token)
        progress = ProgressReporter(
            url=args.progress_url,
            job_id=args.job_id,
            auth_header=auth_header,
        )

        # Run pipeline
        run_segmentation(img3d, args.filename, params, output_options, progress=progress)
    finally:
        print(f"Total workflow runtime: {time.perf_counter() - workflow_start:.2f} s")


if __name__ == "__main__":
    main()
