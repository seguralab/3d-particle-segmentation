# %%
import time
import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import json
from scipy import sparse
from scipy import ndimage as ndi
from scipy.io import savemat, loadmat
from tqdm import tqdm

from skimage.segmentation import watershed
from skimage.transform import resize
from skimage.feature import peak_local_max


# %%
def find_seeds(D, d_peak=np.inf, peak_prom=0):
    ''' Find initial watershed seeds (local maxima) of D. 
        Seeds close to other seeds without significant prominences will be removed.
    Inputs: 
        D (3D numpy.ndarray of float64, shape = (Lx,Ly,Lz)): 
            The distance transform object that watershed will be applied to.
        d_peak (float): Maximum distance considered for close local maxima. 
        peak_prom (float): Minimum prominence required for independent local maximum. 
            For two close local maxima, if the lower local maximum does not have a
            prominence higher than "peak_prom", it will be removed from the seeds.
    Outputs: 
        local_maxi (3D numpy.ndarray of uint16, shape = (Lx,Ly,Lz)): 
            The location and index of seeds. Each nonzero element will be a 
            watershed seed, and its value will be the watershed label.
    '''
    # print('Select seeds:')
    local_maxi_xyz = peak_local_max(D, threshold_abs=3, exclude_border=False, indices = True) # footprint = np.ones((3, 3)), 
    # Coordinates of initial local maxima
    xc = local_maxi_xyz[:,0]
    yc = local_maxi_xyz[:,1]
    zc = local_maxi_xyz[:,2]
    n_local = len(xc)
    ispeak = np.ones(n_local,dtype='bool')
    if n_local > 100:
        range_n_local = tqdm(range(n_local))
    else:
        range_n_local = range(n_local)
    for ii in range_n_local:
        x1 = xc[ii]
        y1 = yc[ii]
        z1 = zc[ii]
        dist2 = (xc-x1)**2 + (yc-y1)**2 + (zc-z1)**2
        # Local maxima close to the select local maximum 
        near = (dist2<d_peak**2).nonzero()[0]
        for jj in near:
            if jj>ii:
                x2 = xc[jj]
                y2 = yc[jj]
                z2 = zc[jj]
                xmin = min(x1,x2)
                xmax = max(x1,x2)
                ymin = min(y1,y2)
                ymax = max(y1,y2)
                zmin = min(z1,z2)
                zmax = max(z1,z2)
                # Find the region between to local maxima, resize it to a cube, and extract the diagonal
                between = D[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]
                leng_resize = max(xmax-xmin+1, ymax-ymin+1, zmax-zmin+1)
                between_resize = resize(between,(leng_resize,leng_resize,leng_resize))
                select = np.arange(leng_resize)
                between_diag = between_resize[select,select,select]

                min_between = between_diag.min()
                min_end = between_diag[[0,-1]].min()
                # The square difference between the minimum of the diagonal and the end point
                sq_dist = np.sqrt(min_end**2 - min_between**2)
                if sq_dist < peak_prom:
                    # Exclude the lower local maximim as seeds
                    if D[x1,y1,z1] < D[x2,y2,z2]:
                        ispeak[ii] = False
                    else:
                        ispeak[jj] = False

    local_maxi = np.zeros(D.shape,dtype = 'uint16')
    index = ispeak.nonzero()[0]
    for (ii,ind) in enumerate(index):
        local_maxi[xc[ind],yc[ind],zc[ind]] = ii+1

    return local_maxi


def refine_seeds(labels, local_maxi, empty_ext=None, s2v_max=1, \
    footprint_direct = np.ones((3,3,3), dtype = 'bool'), \
    footprint_indirect = np.ones((3,3,3), dtype = 'bool')):
    ''' Refine watershed seeds. If a seed generated a volume with large
        surface to area ratio, this seed will be removed.
    Inputs: 
        labels (3D numpy.ndarray of uint16, shape = (Lx,Ly,Lz)): 
            Initial watershed regions. Voxels with the same value (watershed label) 
            belong to the same segmented region.
        local_maxi (3D numpy.ndarray of uint16, shape = (Lx,Ly,Lz)): 
            Initial watershed seeds. Each nonzero element will be a watershed seed, 
            and its value will be the watershed label.
        empty_ext (3D numpy.ndarray of bool, shape = (Lx,Ly,Lz)): 
            Padded regions caused by tilted scanning. Should not be considered as boundaries.
        s2v_max (float): Maximum surface-to-volume ratio of a segmented bead. 
        footprint_direct (3D numpy.ndarray of bool, shape = (3,3,3)):
            footprint indicating direct neighboring voxels
        footprint_indirect (3D numpy.ndarray of bool, shape = (3,3,3)):
            footprint indicating indirect neighboring voxels
    Outputs: 
        local_maxi (3D numpy.ndarray of uint16, shape = (Lx,Ly,Lz)): 
            Initial watershed seeds. Each nonzero element will be a watershed seed, 
            and its value will be the watershed label.
    '''
    # print('Refine seeds:')
    (Lx,Ly,Lz) = labels.shape
    # Coordinates of initial local maxima
    (xc,yc,zc)=local_maxi.nonzero()
    index = local_maxi[xc,yc,zc]
    nlabels = len(xc)
    # Valid data region (not padded from tilted scanning)
    if empty_ext is None:
        data_inter = np.ones((Lx,Ly,Lz), dtype='bool')
    else:
        data_inter = np.logical_not(empty_ext)
    if nlabels > 100:
        range_nlabels = tqdm(range(nlabels))
    else:
        range_nlabels = range(nlabels)
    for ii in range_nlabels:
        # The bead whose label is index[ii]
        bwio = (labels==index[ii])
        (xx,yy,zz) = bwio.nonzero()
        xmin = max(0,xx.min()-1)
        xmax = min(Lx,xx.max()+2)
        ymin = max(0,yy.min()-1)
        ymax = min(Ly,yy.max()+2)
        zmin = max(0,zz.min()-1)
        zmax = min(Lz,zz.max()+2)
        # Crop to a small range near the target region
        bwi = bwio[xmin:xmax, ymin:ymax, zmin:zmax]
        data_interi = data_inter[xmin:xmax, ymin:ymax, zmin:zmax]
        # Number of voxels it contains
        volume = np.logical_and(bwi, data_interi).sum() # volume = bwi.sum()
        if volume>0:
            bwi_boundary_direct = np.logical_xor(bwi,ndi.minimum_filter(bwi,footprint=footprint_direct))
            bwi_boundary_direct = np.logical_and(bwi_boundary_direct, data_interi)
            bwi_boundary_indirect = np.logical_xor(bwi,ndi.minimum_filter(bwi,footprint=footprint_indirect))
            bwi_boundary_indirect = np.logical_and(bwi_boundary_indirect, data_interi)
            # Number of voxels on the boundary. Each indirect boundary voxel counts half.
            surface = bwi_boundary_direct.sum() + (bwi_boundary_indirect.sum()-bwi_boundary_direct.sum())/2
            s2v = surface/volume
            if s2v > s2v_max:
                # If the surface to volume ratio is higher than "s2v_max", this is
                # likely not a valid bead, so it is removed from the seeds.
                local_maxi[xc[ii],yc[ii],zc[ii]] = 0
                # labels[bwio] = 0
    return local_maxi # , labels


def watershed_again(labels, local_maxi, d_peak=np.inf, peak_prom=0, empty_ext=None, s2v_max=1, \
    footprint_direct = np.ones((3,3,3), dtype = 'bool'), \
    footprint_indirect = np.ones((3,3,3), dtype = 'bool')):
    ''' Further cut one region to multiple if possible.
    Inputs: 
        labels (3D numpy.ndarray of uint16, shape = (Lx,Ly,Lz)): 
            Watershed regions. Voxels with the same value (watershed label) 
            belong to the same segmented region.
        local_maxi (3D numpy.ndarray of uint16, shape = (Lx,Ly,Lz)): 
            Watershed seeds. Each nonzero element will be a watershed seed, 
            and its value will be the watershed label.
        d_peak (float): Maximum distance considered for close local maxima. 
        peak_prom (float): Minimum prominence required for independent local maximum. 
            For two close local maxima, if the lower local maximum does not have a
            prominence higher than "peak_prom", it will be removed from the seeds.
        empty_ext (3D numpy.ndarray of bool, shape = (Lx,Ly,Lz)): 
            Padded regions caused by tilted scanning. Should not be considered as boundaries.
        s2v_max (float): Maximum surface-to-volume ratio of a segmented bead. 
        footprint_direct (3D numpy.ndarray of bool, shape = (3,3,3)):
            footprint indicating direct neighboring voxels
        footprint_indirect (3D numpy.ndarray of bool, shape = (3,3,3)):
            footprint indicating indirect neighboring voxels
    Outputs: 
        list_labels (list of 3D numpy.ndarray of bool, shape = (Lx,Ly,Lz)): 
            List of segmented beads. 
    '''
    # print('Watershed again:')
    (Lx,Ly,Lz) = labels.shape
    # Coordinates of initial local maxima
    (xc,yc,zc)=local_maxi.nonzero()
    index = local_maxi[xc,yc,zc]
    n_beads = xc.size
    list_labels = []
    if empty_ext is None:
        empty_ext = np.zeros((Lx,Ly,Lz), dtype='bool')
    for ii in tqdm(range(n_beads)):
        # The bead whose label is index[ii]
        bwio = (labels==index[ii])
        (xx,yy,zz) = bwio.nonzero()
        xmin = max(0,xx.min()-1)
        xmax = min(Lx,xx.max()+2)
        ymin = max(0,yy.min()-1)
        ymax = min(Ly,yy.max()+2)
        zmin = max(0,zz.min()-1)
        zmax = min(Lz,zz.max()+2)
        # Crop to a small range near the target region
        bwi = bwio[xmin:xmax, ymin:ymax, zmin:zmax]
        empty_exti = empty_ext[xmin:xmax, ymin:ymax, zmin:zmax]

        # Go through a similar process as segmenting the full 3D image
        bwi_data = np.logical_or(bwi, empty_exti)
        Di = ndi.distance_transform_edt(bwi_data) * bwi
        local_maxii = find_seeds(Di, d_peak, peak_prom)
        (xci,yci,zci) = local_maxii.nonzero()
        if xci.size == 1: # Cannot cut anymore
            list_labels.append(sparse.csr_matrix(bwio.reshape(Lx*Ly,Lz)))
        else: # Can possibly further cut
            labeli = watershed(-Di, markers=local_maxii, mask=bwi, watershed_line=False, connectivity=2) #, markers
            local_maxiif = refine_seeds(labeli, local_maxii, empty_exti, s2v_max, footprint_direct, footprint_indirect)
            (xci,yci,zci) = local_maxiif.nonzero()
            indexi = local_maxiif[xci,yci,zci]
            n_beadsi = xci.size
            if n_beadsi>1:
                if labeli.max() > n_beadsi: # If some seeds were removed, redo the watershed
                    labeli = watershed(-Di, markers=local_maxiif, mask=bwi, watershed_line=False, connectivity=2) #, markers
                for jj in range(n_beadsi):
                    bwioc = bwio.copy()
                    bwioc[xmin:xmax, ymin:ymax, zmin:zmax] = (labeli == indexi[jj])
                    list_labels.append(sparse.csr_matrix(bwioc.reshape(Lx*Ly,Lz)))
            else: # Cannot cut anymore
                list_labels.append(sparse.csr_matrix(bwio.reshape(Lx*Ly,Lz)))

    return list_labels


def smooth_region(label, footprint_neighbor = np.ones((1,1,1), dtype='uint8')):
    ''' Smooth the bead by removing sharp voxels.
    Inputs: 
        label (3D numpy.ndarray of bool, shape = (Lx,Ly,Lz)): 3D bead.
        footprint_neighbor (3D numpy.ndarray of bool, shape = (3,3,3)):
            footprint indicating neighboring voxels
    Outputs: 
        label (3D numpy.ndarray of bool, shape = (Lx,Ly,Lz)): Smoothed 3D bead.
    '''
    (Lx,Ly,Lz) = label.shape
    (xx,yy,zz)=label.nonzero()
    xmin = max(0,xx.min()-1)
    xmax = min(Lx,xx.max()+2)
    ymin = max(0,yy.min()-1)
    ymax = min(Ly,yy.max()+2)
    zmin = max(0,zz.min()-1)
    zmax = min(Lz,zz.max()+2)
    # Crop to a small range near the target region
    labeli = label[xmin:xmax, ymin:ymax, zmin:zmax]
    n_px = labeli.sum()
    n_px0 = labeli.size
    th_neighbor = np.ceil(footprint_neighbor.sum()/2)
    while n_px0 > n_px: # Until no more voxels can be removed
        n_neighbor = ndi.convolve(labeli.astype('uint8'), footprint_neighbor)
        neighbor = n_neighbor>=th_neighbor
        # Remove sharp voxels: more than a half of neighboring voxels are not within this bead
        labeli = np.logical_and(labeli, neighbor)
        n_px0 = n_px.copy()
        n_px = labeli.sum()
    label[xmin:xmax, ymin:ymax, zmin:zmax] = labeli
    return label


def list_sparse(list_labels_2, shape):
    ''' Convert the 3D bead to a sparse representation.
    Inputs: 
        list_labels (list of 3D numpy.ndarray of bool, shape = (Lx,Ly,Lz)): 
            List of segmented beads. 
    Outputs: 
        list_bead_voxel_count (list of float): 
            List of number of voxels of beads. 
        list_bead_data (list of 2D numpy.ndarray of int converted to list, shape = (n,2)): 
            List of sparse representation of beads. 
    '''
    list_bead_voxel_count = []
    list_bead_data = []
    for label_2 in tqdm(list_labels_2):
        label = label_2.A.reshape(shape)
        # convert 3D array to 1D index according to MATLAB order
        label_1d = label.transpose([2,1,0]).ravel()
        # locate the starting and ending points of the bead
        label_1d_diff = np.diff(np.pad(label_1d.astype('int8'), 1, 'constant', constant_values=0))
        start = (label_1d_diff==1).nonzero()[0]+1
        finish = (label_1d_diff==-1).nonzero()[0]
        bead_data = np.concatenate([[start],[finish]]).T
        list_bead_data.append(bead_data.tolist())
        list_bead_voxel_count.append(int(label.sum()))
    return list_bead_voxel_count, list_bead_data
