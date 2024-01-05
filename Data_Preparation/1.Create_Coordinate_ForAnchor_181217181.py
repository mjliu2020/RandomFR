import os
import numpy as np


def _get_top_left_points(roi_bbox, stride):
    """
    Generate top-left points for bounding boxes

    Parameters
    ----------
    roi_bbox: roi region, defined by [xmin, ymin, zmin, xmax, ymax, zmax]
    stride: stride between adjacent bboxes, a list/tuple containing three digits, defined by (x, y, z)

    Returns
    -------
    tl_x: x coordinates of top-left points, n x 1 numpy array
    tl_y: y coordinates of top-left points, n x 1 numpy array
    tl_z: z coordinates of top-left points, n x 1 numpy array
    """
    xmin, ymin, zmin, xmax, ymax, zmax = roi_bbox
    roi_width = xmax - xmin
    roi_height = ymax - ymin
    roi_length = zmax - zmin

    # get the offset between the first top-left point of patch box and the
    # top-left point of roi_bbox
    offset_x = xmax - np.arange(0, roi_width, stride[0])[-1]
    offset_y = ymax - np.arange(0, roi_height, stride[1])[-1]
    offset_z = zmax - np.arange(0, roi_length, stride[2])[-1]
    offset_x = (offset_x) // 2
    offset_y = (offset_y) // 2
    offset_z = (offset_z) // 2

    # get the coordinates of all top-left points
    tl_x = np.arange(xmin, xmax, stride[0])[:-1] + offset_x
    tl_y = np.arange(ymin, ymax, stride[1])[:-1] + offset_y
    tl_z = np.arange(zmin, zmax, stride[2])[:-1] + offset_z
    tl_x, tl_y, tl_z = np.meshgrid(tl_x, tl_y, tl_z)
    tl_x = np.reshape(tl_x, [-1, 1])
    tl_y = np.reshape(tl_y, [-1, 1])
    tl_z = np.reshape(tl_z, [-1, 1])
    print(tl_x.shape)
    print(tl_y.shape)
    print(tl_z.shape)

    return tl_x, tl_y, tl_z


tl_x, tl_y, tl_z = _get_top_left_points(
    roi_bbox = [0, 0, 0, 181, 217, 181],
    stride = [32, 32, 32]  # Equal to anchor size
    )

save_path = './AnchorMask/FixedCoordinate_forAnchorSize32'
os.makedirs(f'{save_path}', exist_ok=True)
np.save(f'{save_path}/tl_x.npy', tl_x)
np.save(f'{save_path}/tl_y.npy', tl_y)
np.save(f'{save_path}/tl_z.npy', tl_z)
