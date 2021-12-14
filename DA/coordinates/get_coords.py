import numpy as np
import os

lpmt_x = np.load('/home/arsde/J21_SYSU/DA/coordinates/lpmt_x.npz', 'rb')
lpmt_y = np.load('/home/arsde/J21_SYSU/DA/coordinates/lpmt_y.npz', 'rb')
lpmt_z = np.load('/home/arsde/J21_SYSU/DA/coordinates/lpmt_z.npz', 'rb')
R = np.load('/home/arsde/J21_SYSU/DA/coordinates/R.npz')


def get_lpmt_coords(pmtIDs):
  return lpmt_x['arr_0'][pmtIDs], lpmt_y['arr_0'][pmtIDs], lpmt_z['arr_0'][pmtIDs]

def get_R_coords():
  return R['arr_0']

