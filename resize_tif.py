import numpy as np
import tifffile
from matplotlib import pyplot as plt
from PIL import Image, ImageColor
from scipy.ndimage import zoom

#new = LifFile('.\\lif_files\\Single_image_project_100_scaffold.lif')
# dx = 2.275; dy = 2.275; dz = 4.2844; # voxel sizes in the original 3D image (unit: um)
# dxyz = 2; # voxel size in the resized 3D image (unit: um)

def get_pixel_and_z_dimensions(filename):
    with tifffile.TiffFile(filename) as tif:
        xres = yres = zstep = None
        # Extract X and Y resolutions if available
        if 'XResolution' in tif.pages[0].tags:
            xres = tif.pages[0].tags['XResolution'].value
        if 'YResolution' in tif.pages[0].tags:
            yres = tif.pages[0].tags['YResolution'].value

        dx = 1 / (xres[0] / xres[1]) if isinstance(xres, tuple) else 1 / xres if xres else None
        dy = 1 / (yres[0] / yres[1]) if isinstance(yres, tuple) else 1 / yres if yres else None

        # Attempt to find Z step size, might be in a custom tag or description
        for page in tif.pages:
            if 'ImageDescription' in page.tags:
                description = page.tags['ImageDescription'].value
                # You might need to adjust this part to match the specific format in your files
                if "spacing=" in description:
                    start_index = description.index("spacing=") + len("spacing=")
                    end_index = description.index("\n", start_index)
                    zstep = float(description[start_index:end_index])
                else:
                    zstep = dx
            else:
                zstep = dx

        return dx, dy, zstep

def convert_tif(filename, dx, dy, dz, dxyz, fluorescent_label, crop_bool):
    # Read the TIFF file; it's assumed to contain multiple images in a Z-stack
    img_stack = tifffile.imread(filename)

    # Shape of the image stack
    Lz, Lx, Ly = img_stack.shape

    # Create 3-D array of pixel value data
    if fluorescent_label == 1: # beads are labeled
        img3d = np.dstack([z - np.percentile(z, 1) for z in img_stack])
    else: # void space is labeled
        img3d = np.dstack([(z - np.percentile(z, 1))*-1 + 255 for z in img_stack])

    # # Create 3D array of binary pixel data directly
    # if fluorescent_label == 1:  # beads are labeled
    #     img3d = np.dstack([np.reshape(z, (Lx, Ly), order='F') for z in img_stack])
    # else:  # void space is labeled
    #     img3d = np.dstack([255 - np.reshape(z, (Lx, Ly), order='F') for z in img_stack])

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