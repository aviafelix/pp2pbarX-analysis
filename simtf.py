#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib
import argparse
import math
import plotsettings

'''
TODO ! calculate uncertainties !
'''

def histogram(
    data,
    type='window',
    logscale=False,
    # for histogram_means*:
    n_bins=None,
    # for histogram_window*:
    n_points=None,
    window_size=None,
    is_relative_window=True,
    # norm=None,
):
    '''
     Returns histogram of the 'type' type.
     Here is 'data' is Pandas' Series object
     (or DataFrame['index'])

     note: this code is slightly slower
     :::
    # HISTOGRAM_TYPES = {
    #     'means': 'histogram_means',
    #     'discrete': 'histogram_discrete',
    #     'window': 'histogram_window',
    #     'window_2': 'histogram_window_v2',
    # }

    # HISTOGRAM_LOGSCALE_TYPES = {
    #     'means': 'histogram_means_log',
    #     'discrete': 'histogram_discrete_log',
    #     'window': 'histogram_window_log',
    #     'window_2': 'histogram_window_v2_log',
    # }

    # if logscale:
    #     return globals().get(HISTOGRAM_LOGSCALE_TYPES[type])(data)
    # else:
    #     return globals().get(HISTOGRAM_TYPES[type])(data)

    # or:
    # if logscale:
    #     return globals()[HISTOGRAM_LOGSCALE_TYPES[type]](data)
    # else:
    #     return globals()[HISTOGRAM_TYPES[type]](data)
    '''

    HISTOGRAM_TYPES = {
        'means': histogram_means,
        'discrete': histogram_discrete,
        'window': histogram_window,
        'window_2': histogram_window_v2,
    }

    HISTOGRAM_LOGSCALE_TYPES = {
        'means': histogram_means_log,
        'discrete': histogram_discrete_log,
        'window': histogram_window_log,
        'window_2': histogram_window_v2_log,
    }

    if logscale:
        # pass all local variables further:
        # return HISTOGRAM_LOGSCALE_TYPES[type](data)
        return HISTOGRAM_LOGSCALE_TYPES[type](**locals())
    else:
        # pass all local variables further:
        # return HISTOGRAM_TYPES[type](data)
        return HISTOGRAM_TYPES[type](**locals())


def histogram_means(data, n_bins, **kwargs):
    '''
    Calculates histogram by dividing the interval of values
    by `n_bins` subintervals. Linear scale is used.

    `data` is Pandas.Series object.
    `n_bins` is number of intervals.
    # `norm` is normalization.

    and `**kwargs` is for all other unnecessary variables
    '''
    # The total number of values
    n_total = len(data.index)

    if n_total == 0 or n_bins == 0:
        return pd.DataFrame()

    # Min and max values to determine the full interval size:
    data_min = data.min()
    data_max = data.max()

    data_interval = data_max - data_min
    delta = data_interval / n_bins

    hist = []

    # number of points is n_bins + 1
    for i in range(n_bins):
        # coo = data_min + data_interval * (i/n_bins)
        coo = data_min + delta*i

        left_bound = coo
        right_bound = coo + delta

        # extract data values in window [left_bound; right_bound]
        dn = data[
            (data >= left_bound) &
            (data <= right_bound)
        ]

        # add point to the result
        # cnt = dn.count()
        cnt = len(dn.index)
        hist.append((
            coo + 0.5*delta,
            dn.mean(),
            # cnt/n_total,
            cnt,
            cnt/(n_total*delta),
        ))

    return pd.DataFrame(
        hist,
        columns=[
            'coo',
            'mean',
            'n_normalized',
            'n_delta_normalized',
        ],
    )


def histogram_means_log(data, n_bins, **kwargs):
    '''
    Calculates histogram by dividing the interval of values
    by `n_bins` subintervals. Logarithmic scale is used.

    `data` is Pandas.Series object.
    `n_bins` is number of intervals.
    # `norm` is normalization.

    and `**kwargs` is for all other unnecessary variables
    '''
    # The total number of values
    n_total = len(data.index)

    if n_total == 0 or n_bins == 0:
        return pd.DataFrame()

    # Min and max values to determine the full interval size:
    data_min = data.min()
    data_max = data.max()

    log_data_min = math.log(data_min)
    log_data_max = math.log(data_max)

    max_div_min = data_max / data_min
    data_interval = log_data_max - log_data_min
    delta_xi = data_interval / n_bins

    log_data = data.apply(math.log)

    hist = []

    # number of points is n_bins + 1
    for i in range(n_bins):
        # coo = log_data_min + data_interval * (i/n_bins)
        # coo = log_data_min + delta_xi*i

        # left_bound = coo
        # right_bound = coo + delta_xi

        # otherwise values on bounds are not taken into account
        left_bound = log_data_min + data_interval*(i/n_bins)
        right_bound = log_data_max - data_interval*(1.0-(i+1)/n_bins)
        coo = left_bound

        # extract data values in window [left_bound; right_bound]
        dn = data[
            (log_data >= left_bound) &
            (log_data <= right_bound)
            # (data >= math.exp(left_bound)) &
            # (data <= math.exp(right_bound))
        ]

        # required for normalization
        delta = data_min * math.pow(max_div_min, i/n_bins) * (math.pow(max_div_min, 1.0/n_bins) - 1.0)

        # add point to the result
        # cnt = dn.count()
        cnt = len(dn.index)
        hist.append((
            math.exp(coo + 0.5*delta_xi),
            dn.mean(),
            cnt,
            # cnt/n_total,
            cnt/(n_total*delta),
        ))

    return pd.DataFrame(
        hist,
        columns=[
            'coo',
            'mean',
            'n_normalized',
            'n_delta_normalized',
        ],
    )


def histogram_discrete(data, n_bins, **kwargs):
    '''
    Calculates histogram by dividing the interval of values
    by `n_bins` subintervals. Linear scale is used.
    Pregenerated histogram data array is filled by data.

    `data` is Pandas.Series object.
    `n_bins` is number of intervals.
    # `norm` is normalization.

    and `**kwargs` is for all other unnecessary variables
    '''
    # The total number of values
    n_total = len(data.index)

    if n_total == 0 or n_bins == 0:
        return pd.DataFrame()

    # Min and max values to determine the full interval size:
    data_min = data.min()
    data_max = data.max()

    data_interval = data_max - data_min
    delta = data_interval / n_bins

    hist = pd.DataFrame(
        [
            (0.0, ) * 4,
        ] * n_bins,
        columns=[
            'coo',
            'mean',
            'n_normalized',
            'n_delta_normalized',
       ],
    )

    # find indices of bins in histogram for every value
    idx_data = (float(n_bins) * (data - data_min) / (data_max - data_min)).apply(math.floor)

    # idx_data[ idx_data >= n_bins ] = n_bins - 1
    idx_data[ idx_data >= n_bins ] -= 1

    for idx in range(n_bins):
        dn = data[ idx_data == idx ]

        hist['coo'][idx] = data_min + delta*(idx + 0.5)
        hist['mean'][idx] = dn.mean()
        # cnt = len(dn.count())
        cnt = len(dn.index)
        hist['n_normalized'][idx] = cnt
        # hist['n_normalized'][idx] = cnt / n_total
        hist['n_delta_normalized'][idx] = cnt / (delta*n_total)

    return hist


def histogram_discrete_log(data, n_bins, **kwargs):
    '''
    Calculates histogram by dividing the interval of values
    by `n_bins` subintervals. Logarithmic scale is used.
    Pregenerated histogram data array is filled by data.

    `data` is Pandas.Series object.
    `n_bins` is number of intervals.
    # `norm` is normalization.

    and `**kwargs` is for all other unnecessary variables
    '''
    # The total number of values
    n_total = len(data.index)

    if n_total == 0 or n_bins == 0:
        return pd.DataFrame()

    # Min and max values to determine the full interval size:
    data_min = data.min()
    data_max = data.max()

    log_data_min = math.log(data_min)
    log_data_max = math.log(data_max)

    max_div_min = data_max / data_min
    data_interval = log_data_max - log_data_min
    delta_xi = data_interval / n_bins

    hist = pd.DataFrame(
        [
            (0.0, ) * 4,
        ] * n_bins,
        columns=[
            'coo',
            'mean',
            'n_normalized',
            'n_delta_normalized',
        ]
    )

    # find indices of bins in histogram for every value
    idx_data = (float(n_bins) * (data.apply(math.log) - log_data_min) / (log_data_max - log_data_min)).apply(math.floor)

    # idx_data[ idx_data >= n_bins ] = n_bins - 1
    idx_data[ idx_data >= n_bins ] -= 1

    for idx in range(n_bins):
        delta = data_min * math.pow(max_div_min, idx/n_bins) * (math.pow(max_div_min, 1.0/n_bins) - 1.0)

        dn = data[ idx_data == idx ]

        hist['coo'][idx] = math.exp(log_data_min + delta_xi*(idx + 0.5))
        hist['mean'][idx] = dn.mean()
        # cnt = len(dn.count())
        cnt = len(dn.index)
        hist['n_normalized'][idx] = cnt
        # hist['n_normalized'][idx] = cnt / n_total
        hist['n_delta_normalized'][idx] = cnt / (delta*n_total)

    return hist


def histogram_window(data, n_points, window_size, is_relative_window=True, **kwargs):
    '''
    Calculates histogram in `n_points` windows of values. Linear scale
    is used. Pregenerated histogram data array is filled by data.

    `data` is Pandas.Series object.
    `n_points` is number of intervals.
    `window_size` is window width.
    `is_relative_window` is setted to `True` if windows size is
        relative value, otherwise it is absolute value.

    and `**kwargs` is for all other unnecessary variables
    '''
    # The total number of values
    n_total = len(data.index)

    if n_total == 0 or n_points == 0:
        return pd.DataFrame()

    # Min and max values to determine the full interval size:
    data_min = data.min()
    data_max = data.max()

    data_interval = data_max - data_min

    if is_relative_window:
        half_window_size = 0.5 * window_size * data_interval
    else:
        half_window_size = 0.5 * window_size

    delta = data_interval / n_points

    hist = []

    # number of points is n_points+1
    for i in range(n_points+1):
        # h = i / n_points
        # left_bound = (1-h) * data_min + h * data_max - half_window_size
        left_bound = data_min + delta*i - half_window_size
        # h = (i+1) / n_points
        # right_bound = (1-h) * data_min + h * data_max + half_window_size
        right_bound = data_min + delta*i + half_window_size

        if left_bound < data_min:
            left_bound = data_min

        if right_bound > data_max:
            right_bound = data_max

        norm_window_size = right_bound - left_bound

        # extract data values in window [left_bound; right_bound]
        dn = data[
            (data >= left_bound) &
            (data <= right_bound)
        ]

        # add point to the result
        # cnt = dn.count()
        cnt = len(dn.index)
        hist.append((
            0.5 * (right_bound + left_bound),
            dn.mean(),
            # cnt/n_total,
            cnt,
            cnt/(n_total*norm_window_size),
        ))

    return pd.DataFrame(
        hist,
        columns=[
            'coo',
            'mean',
            'n_normalized',
            'n_delta_normalized',
        ],
    )


def histogram_window_v2(data, n_points, window_size, is_relative_window=True, **kwargs):
    '''
    based on `means` function
    '''
    # The total number of values
    n_total = len(data.index)

    if n_total == 0 or n_points == 0:
        return pd.DataFrame()

    # Min and max values to determine the full interval size:
    data_min = data.min()
    data_max = data.max()

    data_interval = data_max - data_min

    if is_relative_window:
        norm_window_size = window_size * data_interval
        half_window_size = 0.5 * norm_window_size
    else:
        norm_window_size = window_size
        half_window_size = 0.5 * window_size

    cmin = data_min + half_window_size
    cmax = data_max - half_window_size
    cinterval = cmax - cmin

    if cinterval < 0:
        return pd.DataFrame()

    hist = []

    for i in range(n_points+1):
        coo = cmin + cinterval * (i/n_points)
        dn = data[
            (data >= coo-half_window_size) &
            (data <= coo+half_window_size)
        ]
        cnt = dn.count()
        hist.append((
            coo,
            dn.mean(),
            # tsn.count()/n_total,
            cnt,
            cnt/(n_total*norm_window_size),
        ))

    return pd.DataFrame(
        hist,
        columns=[
            'coo',
            'mean',
            'n_normalized',
            'n_delta_normalized',
        ],
    )


def histogram_window_log(data, n_points, window_size, **kwargs):
    '''
    Calculates histogram in `n_points` windows of values. Linear scale
    is used. Pregenerated histogram data array is filled by data.

    `data` is Pandas.Series object.
    `n_points` is number of intervals.
    `window_size` is relative window width.

    and `**kwargs` is for all other unnecessary variables
    '''
    # The total number of values
    n_total = len(data.index)

    if n_total == 0 or n_points == 0:
        return pd.DataFrame()

    # Min and max values to determine the full interval size:
    data_min = data.min()
    data_max = data.max()

    log_data_min = math.log(data_min)
    log_data_max = math.log(data_max)

    data_interval = log_data_max - log_data_min
    half_window_size = 0.5 * window_size * data_interval
    delta_xi = data_interval / n_points

    log_data = data.apply(math.log)

    hist = []

    # number of points is n_points+1
    for i in range(n_points+1):
        # h = i / n_points
        # left_bound = (1-h) * log_data_min + h * log_data_max - half_window_size
        left_bound = log_data_min + delta_xi*i - half_window_size

        # h = (i+1) / n_points
        # right_bound = (1-h) * log_data_min + h * log_data_max + half_window_size
        right_bound = log_data_min + delta_xi*i + half_window_size

        if left_bound < log_data_min:
            left_bound = log_data_min

        if right_bound > log_data_max:
            right_bound = log_data_max

        norm_window_size = math.exp(right_bound) - math.exp(left_bound)

        # extract data values in window [left_bound; right_bound]
        dn = data[
            (log_data >= left_bound) &
            (log_data <= right_bound)
        ]

        # add point to the result
        # cnt = dn.count()
        cnt = len(dn.index)
        hist.append((
            math.exp(0.5*(right_bound + left_bound)),
            dn.mean(),
            # cnt/n_total,
            cnt,
            cnt/(n_total*norm_window_size),
        ))

    return pd.DataFrame(
        hist,
        columns=[
            'coo',
            'mean',
            'n_normalized',
            'n_delta_normalized',
        ],
    )


def histogram_window_v2_log(data, n_points, window_size, **kwargs):
    '''
    based on `means` function
    '''
    # The total number of values
    n_total = len(data.index)

    if n_total == 0 or n_points == 0:
        return pd.DataFrame()

    # Min and max values to determine the full interval size:
    data_min = data.min()
    data_max = data.max()

    log_data_min = math.log(data_min)
    log_data_max = math.log(data_max)

    data_interval = log_data_max - log_data_min
    half_window_size = 0.5 * window_size * data_interval
    delta_xi = data_interval / n_points

    log_data = data.apply(math.log)

    cmin = log_data_min + half_window_size
    cmax = log_data_max - half_window_size
    cinterval = cmax - cmin

    if cinterval < 0:
        return pd.DataFrame()

    hist = []

    for i in range(n_points+1):
        coo = cmin + cinterval * (i/n_points)
        norm_window_size = math.exp(coo+half_window_size) - math.exp(coo-half_window_size)
        dn = data[
            (log_data >= coo-half_window_size) &
            (log_data <= coo+half_window_size)
        ]
        cnt = dn.count()
        hist.append((
            math.exp(coo),
            dn.mean(),
            # tsn.count()/n_total,
            cnt,
            cnt/(n_total*norm_window_size),
        ))

    return pd.DataFrame(
        hist,
        columns=[
            'coo',
            'mean',
            'n_normalized',
            'n_delta_normalized',
        ],
    )


def cubic_spline(xi, yi, logscale=False):
    '''
    returns function, interpolated by the cubic spline

    if logscale is setted to True, then use logarithmic scale
    on the x axis.

    '''
    def f(x):
        if x < _x[0]:
            dx = x - _x[1]
            return _y[1] + _b[1]*dx + _c[1]*dx*dx + _d[1]*dx*dx*dx

        if x > _x[-1]:
            dx = x - _x[-1]
            return _y[-1] + _b[-1]*dx + _c[-1]*dx*dx + _d[-1]*dx*dx*dx

        for k in range(1, n):
            if x >= _x[k-1] and x <= _x[k]:
                dx = x - _x[k]
                return _y[k] + _b[k]*dx + _c[k]*dx*dx + _d[k]*dx*dx*dx

    def f_log(x):
        log_x = math.log(x)
        if log_x < _x[0]:
            dx = log_x - _x[1]
            return _y[1] + _b[1]*dx + _c[1]*dx*dx + _d[1]*dx*dx*dx

        if log_x > _x[-1]:
            dx = log_x - _x[-1]
            return _y[-1] + _b[-1]*dx + _c[-1]*dx*dx + _d[-1]*dx*dx*dx

        for k in range(1, n):
            if log_x >= _x[k-1] and log_x <= _x[k]:
                dx = log_x - _x[k]
                return _y[k] + _b[k]*dx + _c[k]*dx*dx + _d[k]*dx*dx*dx

    n = len(xi)

    if n != len(yi):
        return None

    if logscale:
        _x = np.log(xi)
    else:
        _x = xi
    _y = yi

    _h = np.empty(n)
    _l = np.empty(n)
    _delta = np.empty(n)
    _lambda = np.empty(n)

    _b = np.empty(n)
    _c = np.empty(n)
    _d = np.empty(n)

    # I.
    for k in range(1, n):
        _h[k] = _x[k] - _x[k-1]
        _l[k] = (_y[k] - _y[k-1]) / _h[k]

    # II.
    _delta[1] = -0.5 * _h[2] / (_h[1] + _h[2])
    _lambda[1] = 1.5 * (_l[2] - _l[1]) / (_h[1] + _h[2])

    # III.
    for k in range(3, n):
        _delta[k-1] = -_h[k] / (2*_h[k-1] + 2*_h[k] + _h[k-1]*_delta[k-2])
        _lambda[k-1] = (3*_l[k] - 3*_l[k-1] - _h[k-1]*_lambda[k-2]) / \
                       (2*_h[k-1] + 2*_h[k] + _h[k-1]*_delta[k-2])

    # IV.
    _c[0] = 0.0
    _c[-1] = 0.0

    for k in reversed(range(2, n)):
        _c[k-1] = _delta[k-1]*_c[k] + _lambda[k-1]

    # V.
    for k in range(1, n):
        _d[k] = (_c[k] - _c[k-1]) / (3 * _h[k])
        _b[k] = _l[k] + (2*_c[k]*_h[k] + _h[k]*_c[k-1])/3

    del _h, _l, _delta, _lambda

    if logscale:
        return f_log
    else:
        return f


def akima_spline(xi, yi, logscale=False):
    '''
    returns function, interpolated by the akima spline

    p.s. akima spline is the same that of hermite spline
    '''
    def f(x):
        pass

    return f


def catmullrom_spline(xi, yi, logscale=False):
    '''
    returns function, interpolated by the akima spline
    '''
    def f(x):
        pass

    return f


def integrate(f, minvalue, maxvalue, n=32000):
    '''
    Integration f over x in linear scale.
    Values are summarized in n points.
    '''
    delta = maxvalue - minvalue
    dx = delta / n

    S = 0.0

    # Trapeze
    x_curr = minvalue

    for i in range(n):
        x_prev = x_curr
        x_curr = minvalue + (i+1) / n * delta
        x = minvalue + (i+0.5) / n * delta
        S += (0.25 * (f(x_prev) + f(x_curr)) + 0.5*f(x)) * dx

    return S


def logintegrate(f, minvalue, maxvalue, n=32000):
    '''
    Integration f over x in logarithmic scale.
    Values are summarized in n points.
    '''
    exp = math.exp

    ln10 = math.log(10.0)
    xi_min = math.log(minvalue) / ln10
    xi_max = math.log(maxvalue) / ln10

    delta_xi = xi_max - xi_min
    d_xi = delta_xi / n * ln10

    S = 0.0

    # Trapeze
    xi_curr = xi_min

    for i in range(n):
        xi_prev = xi_curr
        xi_curr = xi_min + (i+1) / n * delta_xi
        xi = xi_min + (i+0.5) / n * delta_xi

        S += (
            0.25 * (
                f(exp(xi_prev*ln10)) * exp(xi_prev*ln10) +
                f(exp(xi_curr*ln10)) * exp(xi_curr*ln10)
            )
            + 0.5 * f(exp(xi*ln10)) * exp(xi*ln10)
        ) * d_xi

    return S


def mean_integrate(f, minvalue, maxvalue, n=32000):
    '''
    Integrate f over x in linear scale and calculates mathematical
    expectation (1st momentum) in given interval.
    Values are summarized in n points.
    '''
    delta = maxvalue - minvalue
    dx = delta / n

    S = 0.0
    Sexp = 0.0

    # Trapeze
    x_curr = minvalue

    for i in range(n):
        x_prev = x_curr
        x_curr = minvalue + (i+1) / n * delta
        x = minvalue + (i+0.5) / n * delta

        f_x_prev = 0.25 * f(x_prev)
        f_x_curr = 0.25 * f(x_curr)
        f_x = 0.5 * f(x)

        S += (f_x_prev + f_x_curr + f_x) * dx
        Sexp += (f_x_prev*x_prev + f_x_curr*x_curr + f_x*x) * dx

    return (Sexp/S, S, )


def log_mean_integrate(f, minvalue, maxvalue, n=32000):
    '''
    Integrate f over x in logarithmic scale and calculates mathematical
    expectation (1st momentum) in given interval.
    Values are summarized in n points.
    '''
    exp = math.exp

    ln10 = math.log(10.0)
    xi_min = math.log(minvalue) / ln10
    xi_max = math.log(maxvalue) / ln10

    delta_xi = xi_max - xi_min
    d_xi = delta_xi / n * ln10

    S = 0.0
    Sexp = 0.0

    # Trapeze
    xi_curr = xi_min

    for i in range(n):
        xi_prev = xi_curr
        xi_curr = xi_min + (i+1) / n * delta_xi
        xi = xi_min + (i+0.5) / n * delta_xi

        f_xi_prev = 0.25*f(exp(xi_prev*ln10))
        f_xi_curr = 0.25*f(exp(xi_curr*ln10))
        f_xi = 0.5*f(exp(xi*ln10))

        S += (
            f_xi_prev * exp(xi_prev*ln10) +
            f_xi_curr * exp(xi_curr*ln10) +
            f_xi      * exp(xi*ln10)
        ) * d_xi

        Sexp += (
            f_xi_prev * exp(2.0*xi_prev*ln10) +
            f_xi_curr * exp(2.0*xi_curr*ln10) +
            f_xi      * exp(2.0*xi*ln10)
        ) * d_xi

    return (Sexp/S, S, )


def main():
    pass


if __name__ == '__main__':
    main()
