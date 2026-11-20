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

        # Convert to microns based on ResolutionUnit.
        # TIFF spec: 1=No unit, 2=Inch, 3=Centimeter.
        # tifffile may return int or enum; coerce to int for comparison.
        res_unit_tag = tif.pages[0].tags.get('ResolutionUnit')
        res_unit = int(res_unit_tag.value) if res_unit_tag else 1
        if res_unit == 2:  # inch → µm
            if dx is not None:
                dx *= 25400
            if dy is not None:
                dy *= 25400
        elif res_unit == 3:  # centimeter → µm
            if dx is not None:
                dx *= 10000
            if dy is not None:
                dy *= 10000

        # Attempt to find Z step size from ImageDescription metadata.
        # Try ImageJ "spacing=" first, then OME-XML "PhysicalSizeZ".
        for page in tif.pages:
            if 'ImageDescription' in page.tags:
                description = page.tags['ImageDescription'].value
                if "spacing=" in description:
                    start_index = description.index("spacing=") + len("spacing=")
                    end_index = description.index("\n", start_index)
                    zstep = float(description[start_index:end_index])
                    break

        if zstep is None and tif.ome_metadata:
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                ns = root.tag.split('}')[0] + '}' if '}' in root.tag else ''
                pixels = root.find(f'.//{ns}Pixels')
                if pixels is not None and 'PhysicalSizeZ' in pixels.attrib:
                    zstep = float(pixels.attrib['PhysicalSizeZ'])
            except Exception:
                pass

        return dx, dy, zstep

def convert_tif(filename, dx, dy, dz, dxyz, fluorescent_label, crop_bool):
    # Read the TIFF file; it's assumed to contain multiple images in a Z-stack
    img_stack = tifffile.imread(filename)

    if img_stack.ndim != 3:
        raise ValueError(
            f"TIF file has {img_stack.ndim}D data (shape={img_stack.shape}), "
            "expected 3D single-channel. Multi-channel TIF files are not supported."
        )

    # Shape of the image stack
    Lz, Lx, Ly = img_stack.shape

    # Create 3-D array of pixel value data
    # Step 1: Per-slice floor removal (subtract 1st percentile)
    if fluorescent_label == 1: # beads are labeled
        img3d = np.dstack([z - np.percentile(z, 1) for z in img_stack])
    else: # void space is labeled
        img3d = np.dstack([(z - np.percentile(z, 1))*-1 + 255 for z in img_stack])

    # Step 2: Global percentile stretch — scale so P99 maps to 255.
    # This normalizes intensity across samples with different exposure/gain,
    # making threshold values consistent regardless of acquisition settings.
    img3d = np.clip(img3d, 0, None)
    p99 = np.percentile(img3d, 99)
    if p99 > 0:
        img3d = img3d / p99 * 255

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