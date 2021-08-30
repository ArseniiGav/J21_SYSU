import numpy as np
import pandas as pd 
import numba

PMT_ID_conversion = pd.read_csv('~/J21_SYSU/PMT_ID_conversion.csv')
cd_ids = np.array(PMT_ID_conversion['CdID'])
pmt_ids = np.array(PMT_ID_conversion['PMTID'])

def convert_pmt_ids(input_ids):
    indices = np.where(np.in1d(cd_ids, input_ids))[0]
    return pmt_ids[indices]
