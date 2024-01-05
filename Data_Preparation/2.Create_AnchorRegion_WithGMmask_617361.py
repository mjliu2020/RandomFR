import os
import SimpleITK as sitk
import numpy as np
import pandas as pd


point_path = './AnchorMask/FixedCoordinate_forAnchorSize32'
tl_x = np.load(f'{point_path}/tl_x.npy').squeeze()
tl_y = np.load(f'{point_path}/tl_y.npy').squeeze()
tl_z = np.load(f'{point_path}/tl_z.npy').squeeze()

GMmask = sitk.GetArrayFromImage(sitk.ReadImage('./template/GreyMatterMask_181217181.nii.gz'))

wholefinal = np.zeros((181, 217, 181))
label = 1
patch_size = 32  # AnchorSize
half_size = patch_size // 2
label_index = []

for i in range(150):  # Initial AnchorNum

    whole = np.zeros((181, 217, 181))
    whole[tl_x[i]:min((tl_x[i] + patch_size), 181), tl_y[i]:min((tl_y[i] + patch_size), 217),
    tl_z[i]:min((tl_z[i] + patch_size), 181)] = 1

    patch_remove = np.multiply(whole, GMmask)
    if np.sum(patch_remove) < 100:  # intersection with gray matter is greater than 100 voxels
        continue

    x_index = tl_x[i] + half_size
    y_index = tl_y[i] + half_size
    z_index = tl_z[i] + half_size
    label_index.append([x_index, y_index, z_index, label, np.sum(patch_remove)])
    wholefinal[tl_x[i]:min((tl_x[i] + patch_size), 181), tl_y[i]:min((tl_y[i] + patch_size), 217),
    tl_z[i]:min((tl_z[i] + patch_size), 181)] = label

    label = label + 1

print('AnchorNum: ', label)

label_index_array = np.array(label_index)
# print(label_index_array.shape)

anchor_path = './AnchorMask/FixedCoordinateWithGMmask_forAnchorSize32'
os.makedirs(f"{anchor_path}", exist_ok=True)
np.save(f'{anchor_path}/AnchorPatch_index.npy', label_index_array)
np.save(f'{anchor_path}/AnchorPatch_mask.npy', wholefinal)

df = pd.DataFrame(label_index_array)
df.to_csv(f'{anchor_path}/AnchorPatch_index.csv')

origin_image = sitk.ReadImage('./template/ch2bet.nii')
whole_image = sitk.GetImageFromArray(wholefinal)
whole_image.SetDirection(origin_image.GetDirection())
whole_image.SetOrigin(origin_image.GetOrigin())
whole_image.SetSpacing(origin_image.GetSpacing())

sitk.WriteImage(whole_image, f'{anchor_path}/AnchorPatch_mask_181217181.nii')

volume = sitk.ReadImage(f'{anchor_path}/AnchorPatch_mask_181217181.nii')
reference_volume = sitk.ReadImage(f'./template/BrainMask_05_617361.nii')
resliced_mask = sitk.Resample(volume, referenceImage=reference_volume,
                              transform = sitk.Transform(),
                              interpolator = sitk.sitkNearestNeighbor,
                              defaultPixelValue = 0.0,
                              )

sitk.WriteImage(resliced_mask, f'{anchor_path}/AnchorPatch_mask_{label-1}AnchorNum_617361.nii')