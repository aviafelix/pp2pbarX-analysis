#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib
from matplotlib import rc
import argparse
import numpy as np
import math

font = {
    'family': 'Verdana',
    'weight': 'normal',
}

agg = {
    'path.chunksize': 100000,
}

figure = {
    'dpi': 100,
    # 'figsize': [19.2, 10.8],
    # 'subplot.bottom': 0.06,
    # 'subplot.hspace': 0.2,
    # 'subplot.left': 0.06,
    # 'subplot.right': 0.96,
    # 'subplot.top': 0.94,
    # 'subplot.wspace': 0.2,
}

savefig = {
    'dpi': 100,
}

rc('font', **font)
rc('figure', **figure)
rc('savefig', **savefig)
rc('agg', **agg)

def read_data(filename):

    return pd.read_csv(
        filename,
        sep=';',
        encoding='utf-8',
        error_bad_lines=False,
        # header=None,
    )

# particles masses in GeVs
proton_mass  = 0.938269998 # 0.938272046 GeV/c^2
neutron_mass = 0.93957 # 0.939565379 GeV/c^2

def plot_graphs(df, filename="images/tmp.pdf"):

    fig = plt.figure()

    plt.xscale('log')
    plt.yscale('log')

    ### primary protons
    # count = df['motherA_e'].count()
    count = df.groupby(['collision', 'event'])['motherA_e'].count().count()
    min_E = df['motherA_e'].min()
    max_E = df['motherA_e'].max()
    # df['motherA_e'].hist(bins=10 ** np.linspace(np.log10(min_E), np.log10(max_E), 25), log=True, label='p (prim)', color='#ff0000', )
    df.groupby(['collision', 'event'])['motherA_e'].mean().hist(bins=10 ** np.linspace(np.log10(min_E), np.log10(max_E), 25), log=True, label='p (prim)', color='#ff0000', )
    # 
    # df['motherA_e'].hist(bins=10 ** np.linspace(np.log10(min_E), np.log10(max_E), 25), log=True, label='p (prim)', color='#ff0000', normed=True, )
    # hist, bins = np.histogram(df['motherA_e'], bins=10 ** np.linspace(np.log10(min_E), np.log10(max_E), 25), )
    # plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), label='p (prim)', color='#ff0000', )

    ### protons and neutrons
    # dfp = df[ df['particle_id'] == 2212 ]
    dfp = df[ (df['particle_id'] == 2212) | (df['particle_id'] == 2112) ]
    count = dfp['particle_e'].count()
    min_E = dfp['particle_e'].min()
    max_E = dfp['particle_e'].max()
    dfp['particle_e'].hist(bins=10 ** np.linspace(np.log10(min_E), np.log10(max_E), 75), log=True, label='p', color='#0000ff', )
    # dfp['particle_e'].hist(bins=10 ** np.linspace(np.log10(min_E), np.log10(max_E), 25), log=True, label='p', color='#0000ff', normed=True, )
    # hist, bins = np.histogram(dfp['particle_e'], bins=10 ** np.linspace(np.log10(min_E), np.log10(max_E), 25), )
    # plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), label='p', color='#0000ff', )

    ### antiprotons and antineutrons
    # dfp = df[ df['particle_id'] == -2212 ]
    dfp = df[ (df['particle_id'] == -2212) | (df['particle_id'] == -2112) ]
    count = dfp['particle_e'].count()
    min_E = dfp['particle_e'].min()
    max_E = dfp['particle_e'].max()
    dfp['particle_e'].hist(bins=10 ** np.linspace(np.log10(min_E), np.log10(max_E), 25), log=True, label='pbar', color='#00ff00', )
    # dfp['particle_e'].hist(bins=10 ** np.linspace(np.log10(min_E), np.log10(max_E), 25), log=True, label='pbar', color='#00ff00', normed=True, )
    # hist, bins = np.histogram(dfp['particle_e'], bins=10 ** np.linspace(np.log10(min_E), np.log10(max_E), 25), )
    # plt.bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), label='n', color='#00ff00', )

    # df[ df['particle_id'] == 2212 ]['particle_e'].hist(bins=25, log=True, label='p', )
    # df[ df['particle_id'] == -2212 ]['particle_e'].hist(bins=25, log=True, label='pbar', )

    # plt.xlim([0.7,1000.0])
    plt.xlim([0.001,1000.0])

    # plt.title('Гистограмма распределения протонов и антипротонов по энергиям')
    plt.xlabel('T, [GeV]')
    plt.ylabel('n')
    plt.legend(loc=0)
    plt.close('all')
    fig.savefig(filename)


def options():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-file',
        '-i',
        required=True,
        help='CSV file with data to analyze',
    )

    return parser.parse_args()

def means(ts, column, Npoints, delta):
    if len(ts) == 0:
        return pd.DataFrame()

    cmin = ts.min() + delta
    cmax = ts.max() - delta
    cinterval = cmax - cmin

    Ntotal = ts.count()

    if cinterval < 0:
        return pd.DataFrame()

    r = []

    # Всё-таки будет N+1 точка
    for i in range(Npoints+1):
        coo = cmin + cinterval * (i/Npoints)
        tsn = ts[
            (ts >= coo-delta) &
            (ts <= coo+delta)
        ]
        r.append((
            coo,
            tsn.mean(),
            tsn.count()/Ntotal,
            0.5*tsn.count()/Ntotal/delta,
        ))

    return pd.DataFrame(
        r,
        columns=[
            column,
            column+'_mean',
            column+'_valN',
            column+'_valNE',
        ],
    )

def logmeans(ts, column, Npoints, logdelta=0.1):
    if len(ts) == 0:
        return pd.DataFrame()

    print("Npoints:", Npoints, "; logdelta =", logdelta)

    logdelta = logdelta * math.log(10)

    print("Npoints:", Npoints, "; logdelta =", logdelta)

    cmin = math.log(ts.min()) + logdelta
    #cmax = math.log(ts.max()) - logdelta
    cmax = math.log(ts.max())
    cinterval = cmax - cmin

    print("cmin =", cmin, "; cmax =", cmax)
    print("cinterval =", cmax-cmin)
    print("Ebound_left:", math.exp(cmin), "Ebound_right:", math.exp(cmax))

    Ntotal = ts.count()

    if cinterval < 0:
        return pd.DataFrame()

    r = []

    # Всё-таки будет N+1 точка
    for i in range(Npoints+1):
        coo = cmin + cinterval * (i/Npoints)
        cb_left = math.exp(coo-logdelta)
        #cb_right = math.exp(coo+logdelta)
        cb_right = math.exp(coo)
        tsn = ts[
            (ts >= cb_left) &
            (ts <= cb_right)
        ]

        print(
            "Coo:", math.exp(coo),
            "; cb_left:", cb_left,
            "; cb_right", cb_right,
            "; delta_cb", cb_right-cb_left,
        )

        r.append((
            math.exp(coo),
            tsn.mean(),
            tsn.count()/Ntotal,
            tsn.count()/Ntotal/(cb_right-cb_left),
        ))

    return pd.DataFrame(
        r,
        columns=[
            column,
            column+'_mean',
            column+'_valN',
            column+'_valNE',
        ],
    )

def particles_info(df):
    # primary protons:
    Emin = df['motherA_e'].min()
    Emax = df['motherA_e'].max()

    # cnt_primary_p = df['motherA_e'].count()
    # sum_prim_p_e = df['motherA_e'].sum()
    cnt_primary_p = df.groupby(['collision', 'event'])['motherA_e'].count().count()
    sum_prim_p_e = df.groupby(['collision', 'event'])['motherA_e'].mean().sum()
    mean_prim_p_e = sum_prim_p_e/cnt_primary_p

    dfx = means(
        ts=df.groupby(['collision', 'event'])['motherA_e'].mean(),
        column='motherA_e',
        Npoints=750,
        delta=1.0,
    )

    dfx.to_csv(
        'test.csv',
        index=False,
        float_format='%.10e',
    )

    # primary protons
    print("=======================")
    print("=== Primary protons ===\n")
    print("Emin_k: {Emin:.5f} GeV; Emax_k: {Emax:.5f} GeV"
        .format(
            Emin=Emin-proton_mass,
            Emax=Emax-proton_mass,
        )
    )
    print("Total count:", cnt_primary_p)
    print("Sum E_tot: {Etot:.5f} GeV; sum E_k: {Ek:.5f} GeV"
        .format(
            Etot=sum_prim_p_e,
            Ek=sum_prim_p_e-cnt_primary_p*proton_mass,
        )
    )
    print("Mean energy per particle:")
    print("E_tot/N: {mean_Etot:.5f} GeV; E_k/N: {mean_Ek:.5f} GeV"
        .format(
            mean_Etot=mean_prim_p_e,
            mean_Ek=mean_prim_p_e-proton_mass,
        )
    )
    print("*************************\n")

    # antiprotons
    dfp = df[ df['particle_id'] == -2212 ]['particle_e']

    Emin_pbar = dfp.min()
    Emax_pbar = dfp.max()

    sum_pbar_e = dfp.sum()
    cnt_pbar_e = dfp.count()
    mean_pbar_e = sum_pbar_e/cnt_pbar_e

    print("========================")
    print("=== Sec. antiprotons ===\n")
    print("Emin_k: {Emin:.5f} GeV; Emax_k: {Emax:.5f} GeV"
        .format(
            Emin=Emin_pbar-proton_mass,
            Emax=Emax_pbar-proton_mass,
        )
    )
    print("Total count:", cnt_pbar_e)
    print("Sum E_tot: {Etot:.5f} GeV; sum E_k: {Ek:.5f} GeV"
        .format(
            Etot=sum_pbar_e,
            Ek=sum_pbar_e-cnt_pbar_e*proton_mass,
        )
    )
    print("Mean energy per particle:")
    print("E_tot/N: {mean_Etot:.5f} GeV; E_k/N: {mean_Ek:.5f} GeV"
        .format(
            mean_Etot=mean_pbar_e,
            mean_Ek=mean_pbar_e-proton_mass,
        )
    )
    print("*************************\n")

    # protons
    dfp = df[ df['particle_id'] == 2212 ]['particle_e']

    Emin_p = dfp.min()
    Emax_p = dfp.max()

    sum_p_e = dfp.sum()
    cnt_p_e = dfp.count()
    mean_p_e = sum_p_e/cnt_p_e

    print("====================")
    print("=== Sec. protons ===\n")
    print("Emin_k: {Emin:.5f} GeV; Emax_k: {Emax:.5f} GeV"
        .format(
            Emin=Emin_p-proton_mass,
            Emax=Emax_p-proton_mass,
        )
    )
    print("Total count:", cnt_p_e)
    print("Sum E_tot: {Etot:.5f} GeV; sum E_k: {Ek:.5f} GeV"
        .format(
            Etot=sum_p_e,
            Ek=sum_p_e-cnt_p_e*proton_mass,
        )
    )
    print("Mean energy per particle:")
    print("E_tot/N: {mean_Etot:.5f} GeV; E_k/N: {mean_Ek:.5f} GeV"
        .format(
            mean_Etot=mean_p_e,
            mean_Ek=mean_p_e-proton_mass,
        )
    )
    print("*************************\n")

    # antineutrons
    dfp = df[ df['particle_id'] == -2112 ]['particle_e']

    Emin_nbar = dfp.min()
    Emax_nbar = dfp.max()

    sum_nbar_e = dfp.sum()
    cnt_nbar_e = dfp.count()
    mean_nbar_e = sum_nbar_e/cnt_nbar_e

    print("=========================")
    print("=== Sec. antineutrons ===\n")
    print("Emin_k: {Emin:.5f} GeV; Emax_k: {Emax:.5f} GeV"
        .format(
            Emin=Emin_nbar-neutron_mass,
            Emax=Emax_nbar-neutron_mass,
        )
    )
    print("Total count:", cnt_nbar_e)
    print("Sum E_tot: {Etot:.5f} GeV; sum E_k: {Ek:.5f} GeV"
        .format(
            Etot=sum_nbar_e,
            Ek=sum_nbar_e-cnt_nbar_e*neutron_mass,
        )
    )
    print("Mean energy per particle:")
    print("E_tot/N: {mean_Etot:.5f} GeV; E_k/N: {mean_Ek:.5f} GeV"
        .format(
            mean_Etot=mean_nbar_e,
            mean_Ek=mean_nbar_e-neutron_mass,
        )
    )
    print("*************************\n")

    # neutrons
    dfp = df[ df['particle_id'] == 2112 ]['particle_e']

    Emin_n = dfp.min()
    Emax_n = dfp.max()

    sum_n_e = dfp.sum()
    cnt_n_e = dfp.count()
    mean_n_e = sum_n_e/cnt_n_e

    print("=====================")
    print("=== Sec. neutrons ===\n")
    print("Emin_k: {Emin:.5f} GeV; Emax_k: {Emax:.5f} GeV"
        .format(
            Emin=Emin_n-neutron_mass,
            Emax=Emax_n-neutron_mass,
        )
    )
    print("Total count:", cnt_n_e)
    print("Sum E_tot: {Etot:.5f} GeV; sum E_k: {Ek:.5f} GeV"
        .format(
            Etot=sum_n_e,
            Ek=sum_n_e-cnt_n_e*neutron_mass,
        )
    )
    print("Mean energy per particle:")
    print("E_tot/N: {mean_Etot:.5f} GeV; E_k/N: {mean_Ek:.5f} GeV"
        .format(
            mean_Etot=mean_n_e,
            mean_Ek=mean_n_e-neutron_mass,
        )
    )
    print("*************************\n")

    print("E_min (pbar): {Emin:.5f}; E_max (pbar): {Emax:.5f}"
        .format(
            Emin=Emin_pbar-proton_mass,
            Emax=Emax_pbar-proton_mass,
        )
    )
    print("E_min (p): {Emin:.5f}; E_max (p): {Emax:.5f}"
        .format(
            Emin=Emin_p-proton_mass,
            Emax=Emax_p-proton_mass,
        )
    )
    print("E_min (nbar): {Emin:.5f}; E_max (nbar): {Emax:.5f}"
        .format(
            Emin=Emin_nbar-neutron_mass,
            Emax=Emax_nbar-neutron_mass,
        )
    )
    print("E_min (n): {Emin:.5f}; E_max (n): {Emax:.5f}"
        .format(
            Emin=Emin_n-neutron_mass,
            Emax=Emax_n-neutron_mass,
        )
    )
    print()
    print("Count pbar:", cnt_pbar_e)
    print("Count p:", cnt_p_e)
    print("Count pbar / Count p:", cnt_pbar_e / cnt_p_e)
    print("Sum Etot_pbar / Sum Etot_p:", sum_pbar_e / sum_p_e)
    print("Mean Etot_pbar / Mean Etot_p:", mean_pbar_e / mean_p_e)

    print()

    print("Count nbar:", cnt_nbar_e)
    print("Count n:", cnt_n_e)
    print("Count nbar / Count n:", cnt_nbar_e / cnt_n_e)
    print("Sum Etot_nbar / Sum Etot_n:", sum_nbar_e / sum_n_e)
    print("Mean Etot_nbar / Mean Etot_n:", mean_nbar_e / mean_n_e)

    print()

    print("Count pbar+nbar:", cnt_pbar_e+cnt_nbar_e)
    print("Count p+n:", cnt_p_e+cnt_n_e)
    print("Count (pbar+nbar) / Count (p+n):",
        (cnt_pbar_e+cnt_nbar_e) / (cnt_p_e+cnt_n_e))
    print("Sum Etot_(pbar+nbar) / Sum Etot_(p+n):",
        (sum_pbar_e+sum_nbar_e) / (sum_p_e+sum_n_e))
    print("Mean Etot_(pbar+nbar) / Mean Etot_(p+n):",
        (mean_pbar_e+mean_nbar_e) / (mean_p_e+mean_n_e))
    print("Mean Etot_(pbar+nbar) / Mean Etot_(p+n):",
        df[ (df['particle_id'] == -2212) | \
            (df['particle_id'] == -2112) ]['particle_e'].mean() / \
        df[ (df['particle_id'] == 2212) | \
            (df['particle_id'] == 2112) ]['particle_e'].mean()
    )

    del dfp

if __name__ == '__main__':

    o = options()
    df = read_data(o.input_file)
    unq = df['particle_id'].unique()

    # delta = 1.0
    # Npoints = 750
    # dfx = means(
    #     ts=df['event'],
    #     column='motherA_e',
    #     Npoints=Npoints,
    #     delta=delta,
    # )

    #delta = 0.1
    delta = 0.5
    Npoints = 220
    dfx = logmeans(
        ts=df['event'],
        column='motherA_e',
        Npoints=Npoints,
        logdelta=delta,
    )

    dfx.to_csv(
        'prim_p_log_{Npoints}_{delta:.2f}.csv'
            .format(
                Npoints=Npoints,
                delta=delta,
            ),
        index=False,
        float_format='%.10e',
    )
    quit()

    particles_info(df=df)

    df['motherA_e'] -= proton_mass
    df.ix[(df.particle_id == 2212) | (df.particle_id == -2212), 'particle_e'] -= proton_mass
    df.ix[(df.particle_id == 2112) | (df.particle_id == -2112),'particle_e'] -= neutron_mass

    plot_graphs(df=df, filename=o.input_file[:o.input_file.rfind('/')]+"/images/result.pdf")
