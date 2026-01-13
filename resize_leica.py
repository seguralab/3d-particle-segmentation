import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageColor
from scipy.ndimage import zoom
from readlif.reader import LifFile

#new = LifFile('.\\lif_files\\Single_image_project_100_scaffold.lif')
# dx = 2.275; dy = 2.275; dz = 4.2844; # voxel sizes in the original 3D image (unit: um)
# dxyz = 2; # voxel size in the resized 3D image (unit: um)

def convert_lif(filename, dx, dy, dz, dxyz, channel_num, fluorescent_label, crop_bool):
    new = LifFile(filename)
    # Adjust channel number
    channel_num = channel_num - 1
    # Access a specific image directly
    img_0 = new.get_image(0)
    # Create a list of images using a generator
    img_list = [i for i in new.get_iter_image()]

    # Access a specific item
    img_0.get_frame(z=0, t=0, c=channel_num)
    # Iterate over different items
    frame_list   = [i for i in img_0.get_iter_t(c=channel_num, z=0)]
    z_list = [i for i in img_0.get_iter_z(t=0, c=channel_num)]
    channel_list = [i for i in img_0.get_iter_c(t=0, z=0)]

    [Lx,Ly,Lz] = new.image_list[0]["dims_n"].values()

    # Create 3-D array of pixel value data
    if fluorescent_label == 1: # beads are labeled
        img3d = np.dstack([(np.reshape(z.getdata(), (Lx, Ly), order='F') - np.percentile(z.getdata(), 1))
                            for z in z_list])
    else: # void space is labeled
        img3d = np.dstack([(np.reshape(z.getdata(), (Lx, Ly), order='F') - np.percentile(z.getdata(), 1))*-1 + 255
                           for z in z_list])

    if crop_bool:
        Lx = round(Lx/2)
        Ly = round(Ly/2)
        Lz = round(Lz/2)
        img3d = img3d[Lx:, Ly:, Lz:]

    Lx1 = np.round(Lx*dx/dxyz)
    Ly1 = np.round(Ly*dy/dxyz)
    Lz1 = np.round(Lz*dz/dxyz)
    img3d_resize = zoom(img3d, (Lx1/Lx, Ly1/Ly, Lz1/Lz))

    return img3d_resize
    #print(np.shape(img3d_resize))