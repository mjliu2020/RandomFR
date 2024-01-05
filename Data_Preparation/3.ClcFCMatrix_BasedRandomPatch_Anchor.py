import os
import SimpleITK as sitk
from scipy import io
import numpy as np


def calc_mean_matrix(fmri_path, GMmask_path, anchor_arr_3d):

    fmri_arr = sitk.GetArrayFromImage(sitk.ReadImage(fmri_path))
    GMmask_arr_3d = sitk.GetArrayFromImage(sitk.ReadImage(GMmask_path))

    print('fmri_arr.shape', fmri_arr.shape)
    print('GMmask_arr_3d.shape', GMmask_arr_3d.shape)
    print('anchor_arr_3d.shape', anchor_arr_3d.shape)

    fmri_arr = np.nan_to_num(fmri_arr)
    GMmask_arr_4d = np.expand_dims(GMmask_arr_3d, axis=0)
    anchor_arr_4d = np.expand_dims(anchor_arr_3d, axis=0)
    print('GMmask_arr_4d.shape', GMmask_arr_4d.shape)
    print('anchor_arr_4d.shape', anchor_arr_4d.shape)

    GMmask_arr_4d = GMmask_arr_4d.repeat(fmri_arr.shape[0], axis=0)
    anchor_arr_4d = anchor_arr_4d.repeat(fmri_arr.shape[0], axis=0)
    print('GMmask_arr_4d.shape', GMmask_arr_4d.shape)
    print('anchor_arr_4d.shape', anchor_arr_4d.shape)

    anchorresult = []
    for i in range(1, 113):  # (1, anchorNum + 1)
        anchotmp_part0 = np.where(anchor_arr_4d == i, 1, 0)
        anchotmp_part = np.multiply(GMmask_arr_4d, anchotmp_part0)

        anchoresult_part = np.multiply(fmri_arr, anchotmp_part)

        sumtemp = np.sum(anchotmp_part, axis=(1, 2, 3))
        sum = sumtemp[0]
        anchoresult_mean = np.sum(anchoresult_part, axis=(1, 2, 3)) / sum
        anchorresult.append(anchoresult_mean)

    patch_size = 8  # 16 12 8
    half_size = int(patch_size / 2)
    x_bag = np.random.choice(61, 10000, replace=True)
    y_bag = np.random.choice(73, 10000, replace=True)
    z_bag = np.random.choice(61, 10000, replace=True)

    stopflag = 0
    result = []
    position_add_result = []
    for k in range(10000):

        whole = np.zeros((61, 73, 61))  # zyx

        x_index = x_bag[k]
        y_index = y_bag[k]
        z_index = z_bag[k]
        whole[max(0, (x_index - half_size)):min((x_index + half_size), 61),
        max(0, (y_index - half_size)):min((y_index + half_size), 73),
        max(0, (z_index - half_size)):min((z_index + half_size), 61)] = 1
        whole = np.expand_dims(whole, axis=0)
        whole = whole.repeat(fmri_arr.shape[0], axis=0)

        patch_remove = np.multiply(GMmask_arr_4d, whole)
        if np.sum(patch_remove) < 1:
            print('drop')
            continue
        else:
            stopflag = stopflag + 1
            print('patch:', k)
            result_part = np.multiply(fmri_arr, patch_remove)
            #result_mean = np.mean(result_part, axis=(1, 2, 3))
            sumtemp = np.sum(patch_remove, axis=(1,2,3))
            sum = sumtemp[0]
            result_mean = np.sum(result_part, axis=(1,2,3))/sum

            position = [x_index, y_index, z_index] / np.array([61.0, 73.0, 61.0])
            position_add_result_mean = np.hstack((position, result_mean))
            print('position_add_result_mean.shape:', position_add_result_mean.shape)

            result.append(result_mean)
            position_add_result.append(position_add_result_mean)
        if stopflag == 256:
            break
    anchorresult = np.asarray(anchorresult, np.float64)
    result = np.asarray(result, np.float64)
    position_add_result = np.asarray(position_add_result)
    # print('position_add_result.shape:', position_add_result.shape)
    cc_matrix = np.corrcoef(result, anchorresult)
    # print(cc_matrix.shape)
    return np.nan_to_num(cc_matrix), position_add_result


result_dir = './Result_FCandSignal_BasedPatch_Anchor/NYU_PatchSize8_112AnchorNum'
image_dir = './NYU'
mask_dir = './template'
name_list = os.listdir(f'{image_dir}')
name_list.sort()

anchor_arr_3 = sitk.GetArrayFromImage(sitk.ReadImage('./AnchorMask/FixedCoordinateWithGMmask_forAnchorSize32/AnchorPatch_mask_112AnchorNum_617361.nii'))

sub = 0
for name in name_list:

    print(name, end='  ')

    fmri_path = os.path.join(f'{image_dir}',  f'{name}')
    GMmask_path = os.path.join(f'{mask_dir}', 'GreyMatterMask_617361.nii')

    cc_matrix, result = calc_mean_matrix(fmri_path, GMmask_path, anchor_arr_3)

    print(cc_matrix.shape)
    print(result.shape)
    sub = sub + 1

    result_path = os.path.join(f'{result_dir}/FCMatrix')
    os.makedirs(result_path, exist_ok=True)
    io.savemat(os.path.join(result_path, f'{name[:-7]}.mat'), {'cc_matrix': cc_matrix})
    signal_path = os.path.join(f'{result_dir}/Position_and_ROISignals')
    os.makedirs(signal_path, exist_ok=True)
    io.savemat(os.path.join(signal_path, f'{name[:-7]}.mat'), {'Position_and_ROISignals': result})