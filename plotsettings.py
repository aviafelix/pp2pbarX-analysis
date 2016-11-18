import matplotlib as mpl
from matplotlib import rc

font = {
    # 'family':'FreeMono',
    # 'family': 'Verdana',
    'family':'Liberation Sans',
    'weight': 'normal',
    # 'family':'sans-serif',
    # 'sans-serif':['Helvetica'],
    # 'sans-serif':['Vera Sans'],
}

agg = {
    'path.chunksize': 100000,
}

figure = {
    'dpi': 100,
    # 'figsize': [19.2, 10.8],
    'figsize': [11.00, 6.00],
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
rc('text', usetex=True)
# rc('text', usetex=False)

# mpl.use('SVG')
