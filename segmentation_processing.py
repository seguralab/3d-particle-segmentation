"""
Segmentation processing utilities for 3D particle/bead segmentation.

This module contains helper functions for input detection, loading, parameter
management, and the main segmentation pipeline.
"""

import time
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import json
from scipy import sparse
from scipy import ndimage as ndi
from scipy.io import savemat, loadmat
from tqdm import tqdm

from skimage.segmentation import watershed

from utils import find_seeds, refine_seeds, watershed_again, smooth_region, list_sparse
from resize_tif import convert_tif


def detect_input_type(filename):
    """
    Detect the input file type based on file extension.
    
    Parameters:
    -----------
    filename : str
        Name of the input file
    
    Returns:
    --------
    str
        'tif' for .tif/.tiff files
        'lif' for .lif files
        'mat' for .mat files
    
    Raises:
    -------
    ValueError
        If file type is not supported
    """
    ext = Path(filename).suffix.lower()
    if ext in ['.tif', '.tiff']:
        return 'tif'
    elif ext == '.lif':
        return 'lif'
    elif ext == '.mat':
        return 'mat'
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported types: .tif, .lif, .mat")


def load_and_process_input(filepath, input_type, params):
    """
    Load and process the input image based on its type.
    
    Parameters:
    -----------
    filepath : str
        Full path to the input file
    input_type : str
        Type of input ('tif', 'lif', or 'mat')
    params : dict
        Dictionary containing processing parameters:
        - For TIF: dx, dy, dz, dxyz, fluorescent_label, crop_bool
        - For LIF: dx, dy, dz, dxyz, channel_num, fluorescent_label, crop_bool
        - For MAT: None required (loads img3d_resize from file)
    
    Returns:
    --------
    ndarray
        3D image array
    
    Raises:
    -------
    ValueError
        If input type is unknown
    FileNotFoundError
        If file does not exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if input_type == 'tif':
        dx = params.get('dx', 1.0)
        dy = params.get('dy', 1.0)
        dz = params.get('dz', 1.0)
        dxyz = params.get('dxyz', 1.0)
        fluorescent_label = params.get('fluorescent_label', 1)
        crop_bool = params.get('crop_bool', 0)
        img3d = convert_tif(filepath, dx, dy, dz, dxyz, fluorescent_label, crop_bool)
        return img3d
    
    elif input_type == 'lif':
        from resize_confocal import convert_lif  # Import as needed
        dx = params.get('dx', 0.66)
        dy = params.get('dy', 0.66)
        dz = params.get('dz', 0.1999952)
        dxyz = params.get('dxyz', 0.8)
        channel_num = params.get('channel_num', 1)
        fluorescent_label = params.get('fluorescent_label', 1)
        crop_bool = params.get('crop_bool', False)
        img3d = convert_lif(filepath, dx, dy, dz, dxyz, channel_num, fluorescent_label, crop_bool)
        return img3d
    
    elif input_type == 'mat':
        mat_input = loadmat(filepath)
        img3d = mat_input['img3d_resize']
        return img3d
    
    else:
        raise ValueError(f"Unknown input type: {input_type}")


def get_default_params(input_type):
    """
    Get default processing parameters for each input type.
    
    Parameters:
    -----------
    input_type : str
        Type of input ('tif', 'lif', or 'mat')
    
    Returns:
    --------
    dict
        Default parameters for the given input type
    """
    defaults = {
        'tif': {
            'dx': 1.1375,
            'dy': 1.1375,
            'dz': 1.0,
            'dxyz': 1.5,
            'fluorescent_label': 1,  # 1: beads are labeled, 0: void space is labeled
            'crop_bool': 0,
            'radius_um': 50,          # typical radius of a bead in microns
            's2v_max': 0.65,          # maximum surface-to-volume ratio of a segmented bead
            'th': 150,                # intensity threshold for bead segmentation
            'example_frame': 20,      # frame index for example plot
        },
        'lif': {
            'dx': 0.66,
            'dy': 0.66,
            'dz': 0.1999952,
            'dxyz': 0.8,
            'channel_num': 1,
            'fluorescent_label': 1,
            'crop_bool': False,
            'radius_um': 50,
            's2v_max': 0.65,
            'th': 150,
            'example_frame': 20,
        },
        'mat': {
            'dxyz': 2.0,
            'radius_um': 60,
            's2v_max': 0.5,
            'th': 80,
            'example_frame': 46,
        }
    }
    return defaults.get(input_type, {})


def run_segmentation(img3d, filename, params, output_options):
    """
    Run the full 3D particle segmentation pipeline.
    
    Parameters:
    -----------
    img3d : ndarray
        3D image array
    filename : str
        Name of the input file (used for output naming)
    params : dict
        Segmentation parameters dictionary containing:
        - radius_um, s2v_max, th, example_frame, dxyz
    output_options : dict
        Output control flags containing:
        - does_plot, further_smooth, save_png, save_mat, save_json
    
    Returns:
    --------
    None
        Saves outputs to output/{filename}/ directory
    """
    
    # Extract parameters
    dxyz = params.get('dxyz', 1.0)
    radius_um = params.get('radius_um', 50)
    s2v_max = params.get('s2v_max', 0.65)
    th = params.get('th', 150)
    example_frame = params.get('example_frame', 20)
    
    # Extract output options
    does_plot = output_options.get('does_plot', True)
    further_smooth = output_options.get('further_smooth', True)
    save_png = output_options.get('save_png', True)
    save_mat = output_options.get('save_mat', True)
    save_json = output_options.get('save_json', True)
    
    # Set up output directory
    name = Path(filename).stem
    output_dir = os.path.join('output', name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Derived parameters
    voxel_size = dxyz
    radius = radius_um / voxel_size
    peak_prom = radius / 5
    d_peak = radius
    inten_max = th * 3.33
    th_relative = th / 3
    
    # Output file paths
    file_output = os.path.join(output_dir, f'{name}_segment_{th}.mat')
    file_output_smooth = os.path.join(output_dir, f'Smoothed_{name}_segment_{th}.mat')
    output_json = os.path.join(output_dir, f'{name}_segment_{th}.json')
    
    # Define footprints for morphological operations
    footprint_direct = np.zeros((3, 3, 3), dtype='bool')
    footprint_direct[1, 1, :] = 1
    footprint_direct[:, 1, 1] = 1
    footprint_direct[1, :, 1] = 1
    footprint_indirect = np.ones((3, 3, 3), dtype='bool')
    footprint_neighbor = footprint_direct.astype('uint8').copy()
    footprint_neighbor[1, 1, 1] = 0
    footprint_empty = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], dtype='int16')
    
    # Get image dimensions
    (Lx, Ly, Lz) = dims = img3d.shape
    
    # %% 0. Initial plot
    if does_plot:
        plt.figure()
        plt.imshow(img3d[:, :, example_frame])
        plt.colorbar()
        plt.title("Resized image")
        if save_png:
            plt.savefig(os.path.join(output_dir, f'image_{name}, frame {example_frame}.png'))
        plt.close()
    
    # %% 1. Threshold to binary
    start = time.time()
    img3d_mean3 = ndi.uniform_filter(img3d, 3).astype('int16')
    img3d_mean5 = ndi.uniform_filter(img3d, 5).astype('int16')
    diff_mean = img3d_mean3 - img3d_mean5
    bw = np.logical_or(np.logical_and(img3d > th, diff_mean > -th_relative), img3d > inten_max)
    
    # %% 2. Distance transform
    yz_proj = img3d.max(axis=0)
    empty = yz_proj < 1
    if empty.sum() < Lx:
        empty_ext = np.ones((Lx, Ly, Lz), dtype='bool')
        empty_ext[1:(Lx - 1), 1:(Ly - 1), 1:(Lz - 1)] = 0
    else:
        empty_ext = ndi.maximum_filter(empty, footprint=footprint_empty)
        empty_ext[:, [0, -1]] = 1
        empty_ext = np.repeat(np.expand_dims(empty_ext, axis=0), Lx, axis=0)
        empty_ext[[0, -1], :, :] = 1
    
    bw_data = np.logical_or(bw, empty_ext)
    D = ndi.distance_transform_edt(bw_data) * bw
    
    # %% 3. Find initial watershed seeds
    local_maxi = find_seeds(D, d_peak, peak_prom)
    seed_select = time.time()
    print(f'Seed selection: {seed_select - start:.2f} s')
    
    # %% 4. Initial watershed
    labels = watershed(-D, markers=local_maxi, mask=bw, watershed_line=True, connectivity=2)
    if does_plot:
        plt.figure()
        img = labels[:, :, example_frame].astype('int16')
        num_beads = local_maxi.max()
        img[img <= 0] = 0 - num_beads // 5
        plt.imshow(img)
        plt.colorbar()
        plt.title(f"{num_beads} beads")
        if save_png:
            plt.savefig(os.path.join(output_dir, f'initial_{name}, th={th}, frame {example_frame}.png'))
        plt.close()
    watershed_1 = time.time()
    print(f'Initial watershed: {watershed_1 - seed_select:.2f} s')
    
    # %% 5. Refine watershed seeds
    local_maxi = refine_seeds(labels, local_maxi, empty_ext, s2v_max, footprint_direct, footprint_indirect)
    seed_refine = time.time()
    print(f'Seed refinement: {seed_refine - watershed_1:.2f} s')
    
    # %% 6. Second watershed
    labels = watershed(-D, markers=local_maxi, mask=bw, watershed_line=False, connectivity=2)
    watershed_2 = time.time()
    print(f'Second watershed: {watershed_2 - seed_refine:.2f} s')
    if does_plot:
        plt.figure()
        img = labels[:, :, example_frame].astype('int16')
        num_beads = local_maxi.nonzero()[0].size
        img[img <= 0] = 0 - num_beads // 5
        plt.imshow(img)
        plt.colorbar()
        plt.title(f"{num_beads} beads")
        if save_png:
            plt.savefig(os.path.join(output_dir, f'second_{name}, th={th}, frame {example_frame}.png'))
        plt.close()
    
    # %% 7. Further cut segmented regions
    list_labels_2 = watershed_again(labels, local_maxi, d_peak, peak_prom, empty_ext, s2v_max, footprint_direct, footprint_indirect)
    list_voxel_count = np.array([label_2.sum() for label_2 in list_labels_2])
    order = list_voxel_count.argsort() + 1
    
    labels_2 = np.zeros((Lx * Ly, Lz), dtype='uint16')
    for (ii, ind) in tqdm(enumerate(order), desc='Converting to labeled image'):
        labels_2 += (ind * list_labels_2[ii].A).astype(labels_2.dtype)
    labels = labels_2.reshape((Lx, Ly, Lz))
    
    if save_mat:
        savemat(file_output, {'labels': labels}, do_compression=True)
    
    if does_plot:
        plt.figure()
        img = labels[:, :, example_frame].astype('int16')
        num_beads = labels.max()
        img[img <= 0] = 0 - num_beads // 5
        plt.imshow(img)
        plt.colorbar()
        plt.title(f"{num_beads} beads")
        if save_png:
            plt.savefig(os.path.join(output_dir, f'third_{name}, th={th}, frame {example_frame}.png'))
        plt.close()
    watershed_3 = time.time()
    print(f'Watershed refine: {watershed_3 - watershed_2:.2f} s')
    
    # %% 8. Smooth segmented regions
    if further_smooth:
        for (ii, label_2) in tqdm(enumerate(list_labels_2), desc='Smoothing regions'):
            labels = smooth_region(label_2.A.reshape(Lx, Ly, Lz), footprint_neighbor)
            list_labels_2[ii] = sparse.csr_matrix(labels.reshape(Lx * Ly, Lz))
        
        list_voxel_count = np.array([label_2.sum() for label_2 in list_labels_2])
        order = list_voxel_count.argsort() + 1
        labels_2 = np.zeros((Lx, Ly, Lz), dtype='uint16')
        for (ii, label_2) in tqdm(enumerate(list_labels_2), desc='Converting smoothed labels'):
            labels_2 += (label_2.A.reshape(Lx, Ly, Lz) * (order[ii])).astype(labels_2.dtype)
        labels = labels_2.reshape((Lx, Ly, Lz))
        
        if does_plot:
            plt.figure()
            img = labels[:, :, example_frame].astype('int16')
            num_beads = labels.max()
            img[img <= 0] = 0 - num_beads // 5
            plt.imshow(img)
            plt.colorbar()
            plt.title(f"{num_beads} beads")
            if save_png:
                plt.savefig(os.path.join(output_dir, f'fourth_{name}, th={th}, frame {example_frame}.png'))
            plt.close()
        smooth = time.time()
        print(f'Smooth: {smooth - watershed_3:.2f} s')
        
        if save_mat:
            savemat(file_output_smooth, {'labels': labels}, do_compression=True)
    
    # %% Final: Convert to JSON format
    (list_bead_voxel_count, list_bead_data) = list_sparse(list_labels_2, (Lx, Ly, Lz))
    bead_count = len(list_bead_voxel_count)
    order = order.tolist()
    bead_voxel_count = dict(zip(range(bead_count), [list_bead_voxel_count[x - 1] for x in order]))
    bead_data = dict(zip(range(bead_count), [list_bead_data[x - 1] for x in order]))
    created = time.asctime(time.localtime())
    data_type = "labeled"
    domain_size = [x * voxel_size for x in dims]
    voxel_count = sum(list_bead_voxel_count)
    
    bead_struct = {
        "bead_count": bead_count,
        "bead_voxel_count": bead_voxel_count,
        "created": created,
        "data_type": data_type,
        "domain_size": domain_size,
        "hip_file": filename,
        "voxel_count": int(voxel_count),
        "voxel_size": voxel_size,
        "bead_data": bead_data
    }
    bead_struct_json = json.dumps([bead_struct])
    
    if save_json:
        with open(output_json, 'w') as txt:
            txt.write(bead_struct_json)
    
    print(f"\nSegmentation complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total beads detected: {bead_count}")
