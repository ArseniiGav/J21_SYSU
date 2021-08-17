import ROOT
import pandas as pd
import numpy as np
import os
import sys

def sort_files(files):
    files_sorted = []
    for file in files:
        if 'user' not in file:
             files_sorted.append(file)
    return files_sorted

ind = int(sys.argv[1])
type_of_data = sys.argv[3]
n_files = int(sys.argv[2])

if type_of_data == 'i':
    type_of_data = 'prd04_i'
    option = 'ideal'
elif type_of_data == 'r':
    type_of_data = 'prd04_r'
    option = 'ideal'

path = '/junofs/grid/production/ML/{}/centos7_amd64_gcc830/Pre-Release/J21v1r0-Pre0/positron/uniform/e+_0.0momentums_10.0momentums-extra-params/elecsim_rec/group1'.format(type_of_data)

#df_targets = pd.DataFrame()
files = os.listdir(path)

files = sort_files(files)
files = sorted(files)[ind:ind+n_files]

for i in range(len(files)):
    print('Processed {} file...'.format(i+1))
    edep_array = []
    edepX_array = []
    edepY_array = []
    edepZ_array = []

    f = ROOT.TFile.Open(path+'/'+files[i])
    tracktruth_tree = f.Get("Event/Sim/Truth/TrackElecTruthEvent")
    calib_tree = f.Get("Event/Calib/CalibEvent")

    for evt in tracktruth_tree:
        truths = (evt.TrackElecTruthEvent.truths())
        edep_array.append(truths[0].edep())
        edepX_array.append(truths[0].edepX() / 1000.)
        edepY_array.append(truths[0].edepY() / 1000.)
        edepZ_array.append(truths[0].edepZ() / 1000.)

    npe_arrays = []
    fht_arrays = []
    pmtId_arrays = []

    for evt in calib_tree:
        calibs = (evt.CalibEvent.calibPMTCol())
        npe = []
        fht = []
        pmtId = []
        for calib in calibs:
            npe.append(calib.nPE())
            fht.append(calib.firstHitTime())
            pmtId.append(calib.pmtId())
    
        npe = np.array(npe)
        fht = np.array(fht)
        pmtId = np.array(pmtId)
    
        npe_arrays.append(npe)
        fht_arrays.append(fht)
        pmtId_arrays.append(pmtId)

    npe_arrays = np.array(npe_arrays, dtype=object)
    fht_arrays = np.array(fht_arrays, dtype=object)
    pmtId_arrays = np.array(pmtId_arrays, dtype=object)
    raw_data = np.vstack((pmtId_arrays, npe_arrays, fht_arrays))

    data = pd.DataFrame(data=edep_array)
    data.columns = ['edep']
    data['edepX'] = edepX_array
    data['edepY'] = edepY_array
    data['edepZ'] = edepZ_array

    np.savez_compressed('/junofs/users/gavrikov/{}/train/data/raw_data_train_{}.npz'.format(option, ind+i), a=raw_data)
    data.to_csv('/junofs/users/gavrikov/{}/train/targets/targets_train_{}.csv'.format(option, ind+i), index=False)
    #df_targets = df_targets.append(data)

#df_targets.to_csv('/junofs/users/gavrikov/{}/train/targets/targets_train_{}.csv'.format(option, ind), index=False)