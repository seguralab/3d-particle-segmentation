# %%
import time
import numpy as np
# import cv2
import matplotlib.pyplot as plt
import json
from scipy import sparse
from scipy import ndimage as ndi
from scipy.io import savemat, loadmat
from tqdm import tqdm

from skimage.segmentation import watershed
from skimage.transform import resize
from skimage.feature import peak_local_max

from utils import find_seeds, refine_seeds, watershed_again, smooth_region, list_sparse # , refine_labels


# %%
if __name__ == '__main__':
    # # %% set parameters for SCAPE image
    # name = 'SCAPE_2_norm'
    # # File name of the input data. The version should be MATLAB -v7 or earlier, 
    # # and the variable name should be img3d_resize
    # file_input = '{}_image.mat'.format(name)
    # # voxel size, will be recorded in the json file
    # voxel_size = 2.0
    # # Typical radius of a bead (unit: um). Does not need to be accurate. 
    # radius_um = 60
    # # Typical radius of a bead (unit: voxel).
    # radius = radius_um/voxel_size
    # # For two close local maxima, if the lower local maximum does not have a
    # # prominence higher than "peak_prom", it will be removed from the seeds.
    # peak_prom = 5
    # # Maximum distance considered for close local maxima. 
    # d_peak = radius # 15
    # # Maximum surface-to-volume ratio of a segmented bead
    # s2v_max = 0.5
    # # Absoute intensity threshold. Intensity higher than "th" will be considered 
    # # foreground,unless its relative intensity is below "-th_relative".
    # th = 80
    # # Maximim intensity considered. Intensity higher than "inten_max" will be always 
    # # considered foreground, even if its relative intensity is below "-th_relative".
    # inten_max = 600
    # # Relative threshold. Voxels below "-th_relative" will be considered background, 
    # # even if its absolute intensity is higher than "th".
    # th_relative = 50
    # # Example index of frame to plot (used when does_plot == True)
    # example_frame = 46
    # # Whether further smoothing is applied
    # further_smooth = True
    # # Whether to plot intermediate results
    # does_plot = True
    # # Name of the output mat file before smoothing
    # file_output = '{}_segment_{}.mat'.format(name,th)
    # # Name of the output mat file after smoothing
    # file_output_smooth = 'Smoothed_{}_segment_{}.mat'.format(name,th)
    # # Name of the final output json file
    # output_json = '{}_segment_{}.json'.format(name,th)

    # %% set parameters for confocal image
    name = 'Confocal_2'
    # File name of the input data. The version should be MATLAB -v7 or earlier, 
    # and the variable name should be img3d_resize
    file_input = '{}_image.mat'.format(name)
    # voxel size, will be recorded in the json file
    voxel_size = 2.0
    # Typical radius of a bead (unit: um). Does not need to be accurate. 
    radius_um = 50 # 37 # 
    # Typical radius of a bead (unit: voxel).
    radius = radius_um/voxel_size
    # For two close local maxima, if the lower local maximum does not have a
    # prominence higher than "peak_prom", it will be removed from the seeds.
    peak_prom = radius/5
    # Maximum distance considered for close local maxima. 
    d_peak = radius # 15
    # Maximum surface to volume ratio of a segmented bead
    s2v_max = 0.5
    # Absoute intensity threshold. Intensity higher than "th" will be considered 
    # foreground,unless its relative intensity is below "-th_relative".
    th = 300
    # Maximim intensity considered. Intensity higher than "inten_max" will be always 
    # considered foreground, even if its relative intensity is below "-th_relative".
    inten_max = 1000
    # Relative threshold. Voxels below "-th_relative" will be considered background, 
    # even if its absolute intensity is higher than "th".
    th_relative = 100
    # Example index of frame to plot (used when does_plot == True)
    example_frame = 29
    # Whether further smoothing is applied
    further_smooth = True
    # Whether to plot intermediate results
    does_plot = True
    # Name of the output mat file before smoothing
    file_output = '{}_segment_{}.mat'.format(name,th)
    # Name of the output mat file after smoothing
    file_output_smooth = 'Smoothed_{}_segment_{}.mat'.format(name,th)
    # Name of the final output json file
    output_json = '{}_segment_{}.json'.format(name,th)


    # footprints relating neighboring voxels
    footprint_direct = np.zeros((3,3,3), dtype = 'bool')
    footprint_direct[1,1,:] = 1
    footprint_direct[:,1,1] = 1
    footprint_direct[1,:,1] = 1
    footprint_indirect = np.ones((3,3,3), dtype = 'bool')
    footprint_neighbor = footprint_direct.astype('uint8').copy()
    footprint_neighbor[1,1,1] = 0
    footprint_empty = np.array([[0,0,1,0,0], [0,1,1,1,0], [1,1,1,1,1], [0,1,1,1,0], [0,0,1,0,0]], dtype='int16')

    # %% 0. Load data
    mat_input = loadmat(file_input)
    img3d = mat_input['img3d_resize']
    (Lx,Ly,Lz) = dims = img3d.shape
    if does_plot:
        plt.figure(); plt.imshow(img3d[:,:,example_frame]); plt.colorbar(); plt.title("Resized image"); 
        plt.savefig('image_{}, frame {}.png'.format(name,example_frame)) # plt.show(); # 

    # %% 1. Threshold to binary
    start = time.time()
    img3d_mean3 = ndi.uniform_filter(img3d,3).astype('int16')
    img3d_mean5 = ndi.uniform_filter(img3d,5).astype('int16')
    # relative brightness
    diff_mean = img3d_mean3 - img3d_mean5 
    # binary 3D image after applying three thresholds
    bw = np.logical_or(np.logical_and(img3d > th, diff_mean > -th_relative), img3d > inten_max)

    # %% 2. Distance transform
    # "empty_ext" marks the boundary voxels of the entire image. 
    # These voxels should not be considered boundaries of the beads.
    # In the SCAPE image, some parts of the image are empty because of the tilted scanning. 
    # These parts also should not be considered as boundaries of the beads.
    yz_proj = img3d.max(axis=0)
    empty = yz_proj < 1
    if empty.sum() < Lx:
        empty_ext = np.ones((Lx,Ly,Lz), dtype='bool')
        empty_ext[1:(Lx-1),1:(Ly-1),1:(Lz-1)]=0
    else:
        empty_ext = ndi.maximum_filter(empty, footprint=footprint_empty)
        empty_ext[:,[0,-1]]=1
        empty_ext = np.repeat(np.expand_dims(empty_ext,axis=0),Lx,axis=0)
        empty_ext[[0,-1],:,:]=1

    bw_data = np.logical_or(bw, empty_ext)
    # Distance of a foreground voxel to background. Empty regions are not background that can connected to.
    D = ndi.distance_transform_edt(bw_data) * bw

    # %% 3. Find initial watershed seeds. Seeds close to other seeds 
    # without significant prominences will be removed.
    local_maxi = find_seeds(D, d_peak, peak_prom)
    seed_select = time.time()
    print('Seed selection: {} s'.format(seed_select - start))

    # %% 4. Initial watershed
    labels = watershed(-D, markers=local_maxi, mask=bw, watershed_line=True, connectivity=2)
    if does_plot:
        img = labels[:,:,example_frame].astype('int16')
        num_beads = local_maxi.max()
        img[img<=0] = 0-num_beads//5
        plt.figure(); plt.imshow(img); plt.colorbar(); 
        plt.title("{} beads".format(num_beads)); 
        plt.savefig('initial_{}, th={}, frame {}.png'.format(name,th,example_frame)) # plt.show(); # 
    watershed_1 = time.time()
    print('Initial watershed: {} s'.format(watershed_1 - seed_select))

    # %% 5. Refine watershed seeds. If a seed generated a volume with large
    # surface-to-volume ratio, this seed will be removed.
    local_maxi = refine_seeds(labels, local_maxi, empty_ext, s2v_max, footprint_direct, footprint_indirect)
    seed_refine = time.time()
    print('Seed refinement: {} s'.format(seed_refine - watershed_1))

    # %% 6. Second watershed
    labels = watershed(-D, markers=local_maxi, mask=bw, watershed_line=False, connectivity=2)
    watershed_2 = time.time()
    print('Second watershed: {} s'.format(watershed_2 - seed_refine))
    if does_plot:
        img = labels[:,:,example_frame].astype('int16')
        num_beads = local_maxi.nonzero()[0].size
        img[img<=0] = 0-num_beads//5
        plt.figure(); plt.imshow(img); plt.colorbar(); 
        plt.title("{} beads".format(num_beads)); 
        plt.savefig('second_{}, th={}, frame {}.png'.format(name,th,example_frame)) # plt.show()

    # %% 7. Further cut one segmented region to multiple if possible
    # list of labeled beads
    list_labels_2 = watershed_again(labels, local_maxi, d_peak, peak_prom, empty_ext, s2v_max, \
    footprint_direct, footprint_indirect)
    # list of number of voxels of each bead
    list_voxel_count = np.array([label_2.sum() for label_2 in list_labels_2])
    # Sort the beads according to the number of voxels
    order = list_voxel_count.argsort()+1

    # Convert the list of beads to a labeled 3D image.
    labels_2 = np.zeros((Lx*Ly,Lz), dtype = 'uint16')
    for (ii, ind) in tqdm(enumerate(order)):
        labels_2 += (ind*list_labels_2[ii].A).astype(labels_2.dtype)
    labels = labels_2.reshape((Lx,Ly,Lz))
    savemat(file_output, {'labels':labels}, do_compression=True)
    if does_plot:
        img = labels[:,:,example_frame].astype('int16')
        num_beads = labels.max()
        img[img<=0] = 0-num_beads//5
        plt.figure(); plt.imshow(img); plt.colorbar(); 
        plt.title("{} beads".format(num_beads)); 
        plt.savefig('third_{}, th={}, frame {}.png'.format(name,th,example_frame)) # plt.show()
    watershed_3 = time.time()
    print('Watershed refine: {} s'.format(watershed_3 - watershed_2))

    # %% 8. Smooth the segmented regions by removing sharp voxels.
    if further_smooth:
        for (ii, label_2) in tqdm(enumerate(list_labels_2)):
            labels = smooth_region(label_2.A.reshape(Lx,Ly,Lz), footprint_neighbor)
            list_labels_2[ii] = sparse.csr_matrix(labels.reshape(Lx*Ly,Lz))
        
        list_voxel_count = np.array([label_2.sum() for label_2 in list_labels_2])
        order = list_voxel_count.argsort()+1
        # Convert the list of beads to a labeled 3D image.
        labels_2 = np.zeros((Lx,Ly,Lz),dtype = 'uint16')
        for (ii, label_2) in tqdm(enumerate(list_labels_2)):
            labels_2 += (label_2.A.reshape(Lx,Ly,Lz)*(order[ii])).astype(labels_2.dtype)
        labels = labels_2.reshape((Lx,Ly,Lz))
        if does_plot:
            img = labels[:,:,example_frame].astype('int16')
            num_beads = labels.max()
            img[img<=0] = 0-num_beads//5
            plt.figure(); plt.imshow(img); plt.colorbar(); 
            plt.title("{} beads".format(num_beads)); 
            plt.savefig('fourth_{}, th={}, frame {}.png'.format(name,th,example_frame)) # plt.show()
        smooth = time.time()
        print('Smooth: {} s'.format(smooth - watershed_3))
        savemat(file_output_smooth, {'labels':labels}, do_compression=True)

    # %% Final: convert to json format
    (list_bead_voxel_count, list_bead_data) = list_sparse(list_labels_2, (Lx,Ly,Lz))
    bead_count = len(list_bead_voxel_count)
    order = order.tolist()
    bead_voxel_count = dict(zip(range(bead_count), [list_bead_voxel_count[x-1] for x in order]))
    bead_data = dict(zip(range(bead_count), [list_bead_data[x-1] for x in order]))
    created = time.asctime(time.localtime()) # The format of data and time in python is different from MATLAB
    data_type = "labeled"
    domain_size = [x*voxel_size for x in dims]
    hip_file = file_input
    voxel_count = sum(list_bead_voxel_count)
    bead_struct = {"bead_count": bead_count, "bead_voxel_count": bead_voxel_count, \
        "created": created, "data_type": data_type, "domain_size": domain_size, \
        "hip_file": hip_file, "voxel_count": int(voxel_count), \
        "voxel_size": voxel_size, "bead_data": bead_data}
    bead_struct_json = json.dumps([bead_struct])
    txt=open(output_json,'w')
    txt.write(bead_struct_json)
    txt.close()

# %%
