import os
from glob import glob
import pandas as pd
from scipy.io import loadmat
import numpy as np


data_fold = pd.read_csv(f'./NYU171.csv', skip_blank_lines=True)

for index, row in data_fold.iterrows():

    name = row['SUB_ID']
    label = row['DX_GROUP']-1
    FCMatrix_path = glob(f'../../Data_Preparation/Result_FCandSignal_BasedPatch_Anchor/NYU_PatchSize8_112AnchorNum/FCMatrix/NYU_00{name}_func_preproc.mat')
    Posi_Signal_path = glob(f'../../Data_Preparation/Result_FCandSignal_BasedPatch_Anchor/NYU_PatchSize8_112AnchorNum/Position_and_ROISignals/NYU_00{name}_func_preproc.mat')

    FCMatrix = loadmat(FCMatrix_path[0])['cc_matrix']
    Posi_Signal = loadmat(Posi_Signal_path[0])['Position_and_ROISignals']

    print(name, label, end=' ')
    print(Posi_Signal.shape)

    FCMatrixfinal = FCMatrix[:256, 256:]
    print('FCMatrixfinal_shape: ', FCMatrixfinal.shape)

    posi = Posi_Signal[:, :3]

    os.makedirs('subjectwise_position_new', exist_ok=True)
    np.save(os.path.join('subjectwise_position_new', f'{name}.npy'), posi)
    os.makedirs(f'./FCmatrix_new/{name}', exist_ok=True)
    np.save(os.path.join('./FCmatrix_new', str(name), f'fcmatrix.npy'), FCMatrixfinal)
