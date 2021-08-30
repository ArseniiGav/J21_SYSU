import pandas as pd
import numpy as np
import os
from convert_pmt_ids import *
import re
import sys
from scipy.stats import skew, kurtosis

_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

def sort_files(files):
    files_sorted = []
    for file in files:
        if 'einit' not in file:
             files_sorted.append(file)
    return files_sorted

PMTPos_CD_LPMT = pd.read_csv('~/J21_SYSU/PMTPos_CD_LPMT.csv')

option = sys.argv[1]

if option == 'r':
    option = 'real'
elif option == 'i':
    option = 'ideal'

data_type = sys.argv[2]

if data_type=='tt':
    data_type = 'test/'
    size = 11
    title = "Energies processing..."  
elif data_type=='tn':
    data_type = 'train/'
    energy = ''
    size = 1
    title = "Train data processing..."

# thr_array = [1 + i for i in range(9)] + \
#             [5 * (i + 1) for i in range(1, 18)] + \
#             [91 + i for i in range(9)]

thr_array = np.linspace(0, 95, 20)
thr_array[0] = 2

b1 = int(sys.argv[3])
b2 = int(sys.argv[4])

eps = 50
for en_value in range(size):
    
    if data_type=='test/':
        energy = '/e+_{}/'.format(en_value)

    #path = 'raw_data/{}/'.format(option)+data_type
    path = '/mnt/cephfs/ml_data/mc_2021/raw_data/{}/'.format(option)+data_type

    files_data = os.listdir(path+'data'+energy)
    files_data.sort(key=natural_sort_key)

    files_targets = os.listdir(path+'targets'+energy)
    files_targets = sort_files(files_targets)
    files_targets.sort(key=natural_sort_key)

    df = pd.DataFrame()
    for k in range(len(files_data[b1:b2])):   
        data = np.load(path+'data/'+energy+files_data[k+b1], allow_pickle=True)['a']

        NEvents = data.shape[1]
        for i in range(NEvents):
            nonzeros_inds = np.where(data[2, :][i] != 0.0)[0]
            data[0, :][i] = convert_pmt_ids(data[0, :][i])
            data[0, :][i] = data[0, :][i][nonzeros_inds]
            data[1, :][i] = data[1, :][i][nonzeros_inds]
            data[2, :][i] = data[2, :][i][nonzeros_inds]

        lpmt_charge = data[1, :]
        lpmt_fht = data[2, :]
        
        ht_mean = [lpmt_fht[i].mean() for i in range(NEvents)]
        ht_std = [lpmt_fht[i].std() for i in range(NEvents)]
        ht_skew = [skew(lpmt_fht[i]) for i in range(NEvents)]
        ht_kurtosis = [kurtosis(lpmt_fht[i]) for i in range(NEvents)]

        pe_mean = [lpmt_charge[i].mean() for i in range(NEvents)]
        pe_std = [lpmt_charge[i].std() for i in range(NEvents)]
        pe_skew = [skew(lpmt_charge[i]) for i in range(NEvents)]
        pe_kurtosis = [kurtosis(lpmt_charge[i]) for i in range(NEvents)]

        x_cc = np.zeros(NEvents)
        y_cc = np.zeros(NEvents)
        z_cc = np.zeros(NEvents)
        
        x_cht = np.zeros(NEvents)
        y_cht = np.zeros(NEvents)
        z_cht = np.zeros(NEvents)
        
        accum_charge = np.zeros(NEvents)
        nPMTs = np.zeros(NEvents)

        for i in range(NEvents):
            lpmt_x = np.array(PMTPos_CD_LPMT['x'][data[0, :][i]]) / 1000.
            lpmt_y = np.array(PMTPos_CD_LPMT['y'][data[0, :][i]]) / 1000.
            lpmt_z = np.array(PMTPos_CD_LPMT['z'][data[0, :][i]]) / 1000.

            x_cc[i] = np.sum(lpmt_x * lpmt_charge[i]) / np.sum(lpmt_charge[i])
            y_cc[i] = np.sum(lpmt_y * lpmt_charge[i]) / np.sum(lpmt_charge[i])
            z_cc[i] = np.sum(lpmt_z * lpmt_charge[i]) / np.sum(lpmt_charge[i])
            
            x_cht[i] = np.sum(lpmt_x / (lpmt_fht[i] + eps)) / np.sum(1 / (lpmt_fht[i] + eps))
            y_cht[i] = np.sum(lpmt_y / (lpmt_fht[i] + eps)) / np.sum(1 / (lpmt_fht[i] + eps))
            z_cht[i] = np.sum(lpmt_z / (lpmt_fht[i] + eps)) / np.sum(1 / (lpmt_fht[i] + eps))
            
            accum_charge[i] = np.sum(lpmt_charge[i])
            nPMTs[i] = int(lpmt_charge[i].shape[0])
        
        R_cc = (x_cc**2 + y_cc**2 + z_cc**2)**0.5
        pho_cc = (x_cc**2 + y_cc**2)**0.5
        theta_cc = np.arctan2((x_cc**2 + y_cc**2)**0.5, z_cc)
        phi_cc = np.arctan2(y_cc, x_cc)
        gamma_z_cc = z_cc / (x_cc**2 + y_cc**2)**0.5
        gamma_y_cc = y_cc / (z_cc**2 + x_cc**2)**0.5
        gamma_x_cc = x_cc / (z_cc**2 + y_cc**2)**0.5
        jacob_cc = R_cc**2 * np.sin(theta_cc)

        R_cht = (x_cht**2 + y_cht**2 + z_cht**2)**0.5
        pho_cht = (x_cht**2 + y_cht**2)**0.5
        theta_cht = np.arctan2((x_cht**2 + y_cht**2)**0.5, z_cht)
        phi_cht = np.arctan2(y_cht, x_cht)
        gamma_z_cht = z_cht / (x_cht**2 + y_cht**2)**0.5
        gamma_y_cht = y_cht / (z_cht**2 + x_cht**2)**0.5
        gamma_x_cht = x_cht / (z_cht**2 + y_cht**2)**0.5
        jacob_cht = R_cht**2 * np.sin(theta_cht)

        ht_ps = []
        pe_ps = []
        for thr in thr_array:
            ht_ps.append([np.percentile(data[2, :][i], thr) for i in range(NEvents)])
            pe_ps.append([np.percentile(data[1, :][i], thr) for i in range(NEvents)])
        ht_ps = np.array(ht_ps)
        
        diff_ht_ps = []
        for i in range(1, len(ht_ps)):
                diff_ht_ps.append(ht_ps[i] - ht_ps[i-1])

        features_df = pd.DataFrame()
        features_df['AccumCharge'] = accum_charge
        features_df['nPMTs'] = nPMTs

        features_df['R_cc'] = R_cc
        features_df['pho_cc'] = pho_cc
        features_df['x_cc'] = x_cc
        features_df['y_cc'] = y_cc
        features_df['z_cc'] = z_cc
        features_df['gamma_z_cc'] = gamma_z_cc
        features_df['gamma_y_cc'] = gamma_y_cc
        features_df['gamma_x_cc'] = gamma_x_cc
        features_df['theta_cc'] = theta_cc
        features_df['phi_cc'] = phi_cc
        features_df['jacob_cc'] = jacob_cc

        features_df['R_cht'] = R_cht
        features_df['pho_cht'] = pho_cht
        features_df['x_cht'] = x_cht
        features_df['y_cht'] = y_cht
        features_df['z_cht'] = z_cht
        features_df['gamma_z_cht'] = gamma_z_cht
        features_df['gamma_y_cht'] = gamma_y_cht
        features_df['gamma_x_cht'] = gamma_x_cht
        features_df['theta_cht'] = theta_cht
        features_df['phi_cht'] = phi_cht
        features_df['jacob_cht'] = jacob_cht

        features_df['ht_std'] = ht_std
        features_df['ht_mean'] = ht_mean
        features_df['ht_skew'] = ht_skew
        features_df['ht_kurtosis'] = ht_kurtosis

        features_df['pe_std'] = pe_std
        features_df['pe_mean'] = pe_mean
        features_df['pe_skew'] = pe_skew
        features_df['pe_kurtosis'] = pe_kurtosis

        features_df['ht_2p'] = ht_ps[0]
        features_df['ht_5p'] = ht_ps[1]
        features_df['ht_10p'] = ht_ps[2]
        features_df['ht_15p'] = ht_ps[3]
        features_df['ht_20p'] = ht_ps[4]
        features_df['ht_25p'] = ht_ps[5]
        features_df['ht_30p'] = ht_ps[6]
        features_df['ht_35p'] = ht_ps[7]
        features_df['ht_40p'] = ht_ps[8]
        features_df['ht_45p'] = ht_ps[9]
        features_df['ht_50p'] = ht_ps[10]
        features_df['ht_55p'] = ht_ps[11]
        features_df['ht_60p'] = ht_ps[12]
        features_df['ht_65p'] = ht_ps[13]
        features_df['ht_70p'] = ht_ps[14]
        features_df['ht_75p'] = ht_ps[15]
        features_df['ht_80p'] = ht_ps[16]
        features_df['ht_85p'] = ht_ps[17]
        features_df['ht_90p'] = ht_ps[18]
        features_df['ht_95p'] = ht_ps[19]

        features_df['ht_5-2p'] = diff_ht_ps[0]
        features_df['ht_10-5p'] = diff_ht_ps[1]
        features_df['ht_15-10p'] = diff_ht_ps[2]
        features_df['ht_20-15p'] = diff_ht_ps[3]
        features_df['ht_25-20p'] = diff_ht_ps[4]
        features_df['ht_30-25p'] = diff_ht_ps[5]
        features_df['ht_35-30p'] = diff_ht_ps[6]
        features_df['ht_40-35p'] = diff_ht_ps[7]
        features_df['ht_45-40p'] = diff_ht_ps[8]
        features_df['ht_50-45p'] = diff_ht_ps[9]
        features_df['ht_55-50p'] = diff_ht_ps[10]
        features_df['ht_60-55p'] = diff_ht_ps[11]
        features_df['ht_65-60p'] = diff_ht_ps[12]
        features_df['ht_70-65p'] = diff_ht_ps[13]
        features_df['ht_75-70p'] = diff_ht_ps[14]
        features_df['ht_80-75p'] = diff_ht_ps[15]
        features_df['ht_85-80p'] = diff_ht_ps[16]
        features_df['ht_90-85p'] = diff_ht_ps[17]
        features_df['ht_95-90p'] = diff_ht_ps[18]
        
        features_df['pe_2p'] = pe_ps[0]
        features_df['pe_5p'] = pe_ps[1]
        features_df['pe_10p'] = pe_ps[2]
        features_df['pe_15p'] = pe_ps[3]
        features_df['pe_20p'] = pe_ps[4]
        features_df['pe_25p'] = pe_ps[5]
        features_df['pe_30p'] = pe_ps[6]
        features_df['pe_35p'] = pe_ps[7]
        features_df['pe_40p'] = pe_ps[8]
        features_df['pe_45p'] = pe_ps[9]
        features_df['pe_50p'] = pe_ps[10]
        features_df['pe_55p'] = pe_ps[11]
        features_df['pe_60p'] = pe_ps[12]
        features_df['pe_65p'] = pe_ps[13]
        features_df['pe_70p'] = pe_ps[14]
        features_df['pe_75p'] = pe_ps[15]
        features_df['pe_80p'] = pe_ps[16]
        features_df['pe_85p'] = pe_ps[17]
        features_df['pe_90p'] = pe_ps[18]
        features_df['pe_95p'] = pe_ps[19]
        
        targets = pd.read_csv(path+'targets/'+energy+files_targets[k+b1]) 
        targets['edepR'] = (targets['edepX']**2 + targets['edepY']**2 + targets['edepZ']**2)**0.5
        features_df = pd.concat([features_df, targets], axis=1)
        df = df.append(features_df)
        
    if data_type=='test/':
        df.to_csv('/mnt/cephfs/ml_data/mc_2021/processed_data/ProcessedTest{}/{}MeV.csv.gz'.format(option.capitalize(), en_value), index=False, compression='gzip')
    elif data_type=='train/':
        df.to_csv('/mnt/cephfs/ml_data/mc_2021/processed_data/ProcessedTrain{}/ProcessedTrain_{}.csv.gz'.format(option.capitalize(), str(b1)), index=False, compression='gzip')