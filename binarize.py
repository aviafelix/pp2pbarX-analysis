#!/usr/bin/env python3
import numpy as np

DATADIR = 'pythia-pp2pbarx/examples/results/'

TASKS = [
    # '001', '002', '003', 
    '004', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', 
]

FILES = [
    'p_prim', 'p', 'n', 'pbar', 'nbar',
]

def binarize():
    for taskname in TASKS:
        workdir = DATADIR+taskname+'/'

        if taskname in ('001', '002', '003', ):
            data_prefx = 'WH03_lin_'
        else:
            data_prefx = ''

        for filename in FILES:
            if filename == 'p_prim':
                coln = 2
            else:
                coln = 4
            dat2bin(workdir+filename+'/'+data_prefx+filename, coln=coln)

def dat2bin(filename, coln):
    # data = np.genfromtxt(filename+'.dat', delimiter=';', names=True)['particle_e_kin']
    data = np.loadtxt(filename+'.dat', delimiter=';', skiprows=1, usecols=(coln, ), )
    data.tofile(filename+'.bin')
    del data

if __name__ == '__main__':
    binarize()
