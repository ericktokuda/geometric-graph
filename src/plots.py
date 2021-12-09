#!/usr/bin/env python3
"""Plots """

import argparse
import time, datetime
import os
from os.path import join as pjoin
import inspect

import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from myutils import info, create_readme
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit, leastsq

##########################################################
W = 1200; H = 960

##########################################################
def get_dir_minmax(losses, ind0, optm='max'):
    """Get the neighbouring pixel which corresponds to the minimum/maximum gradient
    magnitude"""
    ind0 = np.array(ind0)
    curloss = losses[ind0[0], ind0[1]]
    
    # x0, y0 = ind
    dirs_ = np.array([
        [+ 1,   0 ], [  0, + 1 ],
        [- 1,   0 ], [  0, - 1 ], ])

    refdiff = 99999999 if optm == 'min' else 0
    refdir = ind0
    h, w = losses.shape

    for dir_ in dirs_:
        i, j = np.array(ind0 + dir_)
        if (i < 0) or (i >= h) or (j < 0) or (j >= w): continue
        curdiff = losses[i, j] - curloss
        if (optm == 'min' and curdiff < refdiff) or (optm == 'max' and curdiff > refdiff):
            refdir = dir_
            refdiff = curdiff
    return refdir

##########################################################
def get_dir(losses, ind, method='greedy'):
    """Choose the 8-connected direction based on neighbourhood given that we are
    trying to minimize the losses"""
    if method == 'greedy':
        return get_dir_minmax(losses, ind, 'min')
    elif method == 'opposmax':
        return - get_dir_minmax(losses, ind, 'max')

##########################################################
def gradient_descent(losses, ind0, lr0):
    errthresh = 1e-3 # Error threshold
    maxsteps = 1000
    curerr = 99999
    lr = lr0
    losstgt = 0 # Target loss

    h, w = losses.shape

    visitted = [] # Store the descent path
    step = 0
    ind = ind0
    while (step < maxsteps) and (curerr > errthresh):
        lr *= .8
        indflat = np.ravel_multi_index(ind, (h, w))
        # if indflat in visitted: break # It is loop, unless the learning rate is different
        visitted.append(indflat)

        # ind = np.unravel_index(indflat, (h, w))

        ind = ind + lr * get_dir(losses, ind, 'opposmax')
        ind = np.array([int(np.round(ind[0])), int(np.round(ind[1]))])
        

        if ind[0] < 0: ind[0] = 0  # Clip values in the borders
        elif ind[0] > h: ind[0] = h - 1
        if ind[1] < 0: ind[1] = 0
        elif ind[1] >= w: ind[1] = w - 1

        curloss = losses[ind[0], ind[1]]
        curerr = np.abs(losstgt - curloss)
        print(curloss)

        newindflat = np.ravel_multi_index(ind, (h, w))
        if newindflat == indflat: break

        step += 1

    return ind, np.unravel_index(visitted, (h, w))

##########################################################
def plot_features_across_time(df, feats, outdir):
    """Plot feature across time for each feature"""
    info(inspect.stack()[0][3] + '()')
    spl = 10 # Plot every spl points
    for feat in feats:
        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        xs = np.unique(df.step)[::spl]
        for a in sorted(np.unique(df.alpha)):
            df2 = df.loc[df.alpha == a]
            ys = df2.groupby(['step'])[feat].mean()[::spl]
            yerr = df2.groupby(['step'])[feat].std()[::spl]
            ax.plot(xs, np.power(ys, 3), alpha=.5, label=a)
            ax.errorbar(xs, ys, yerr, alpha=.5, label=a)
            ax.set_ylabel(feat.capitalize())
            ax.set_xlabel('Time')
        outpath = pjoin(outdir, feat + '.pdf')
        plt.legend()
        plt.savefig(outpath); plt.close()

##########################################################
def print_features_latex(df, feats):
    """Output the features, as requested by Luc"""
    info(inspect.stack()[0][3] + '()')
    spl = 300
    for feat in feats:
        data = []
        xs = np.unique(df.step)[::spl]
        for a in np.unique(df.alpha):
            df2 = df.loc[df.alpha == a]
            ys = df2.groupby(['step'])[feat].mean()[::spl]
            data.append(ys)
        data = np.array(data).T # cols:alphas, rows:time
        df3 = pd.DataFrame(data, columns=[str(a) for a in np.unique(df.alpha)])
        print(df3.to_latex())

##########################################################
def get_average_values(df, feats, spl):
    data = []
    for feat in feats:
        xs = np.unique(df.step)[::spl]
        d = []
        for a in sorted(np.unique(df.alpha)):
            df2 = df.loc[df.alpha == a]
            ys = df2.groupby(['step'])[feat].mean()[::spl]
            d.append(ys)
        data.append(np.array(d).T)
    smpvals = {'alpha' : np.unique(df.alpha),
            'step': xs}
    return np.array(data), smpvals

##########################################################
def get_loss(data, tgt, outdir):
    """Plot loss given by least squared difference"""
    info(inspect.stack()[0][3] + '()')

    losses = np.zeros((data.shape[1], data.shape[2]))
    for i in range(data.shape[0]):
        losses += np.power(data[i, :, :] - tgt[i], 2)

    # Plot loss map
    loss = losses[::300]
    scaler = MinMaxScaler((0, 255))
    normloss = scaler.fit_transform(loss).astype(int)
    plt.imshow(normloss)
    plt.colorbar()
    plt.savefig(pjoin(outdir, 'loss.png'))
    return losses

##########################################################
def get_min(losses, smpvals, outdir):
    """Get the minimum from the @losses matrix"""
    info(inspect.stack()[0][3] + '()')
    h, w = losses.shape
    minind = np.unravel_index(np.argmin(losses), losses.shape)
    steps = smpvals['step']
    alphas = smpvals['alpha']
    stepgt = steps[minind[0]]
    alphagt = alphas[minind[1]]
    print('GT: step, alpha, ind, loss:', stepgt, alphagt, minind,
            losses[minind[0], minind[1]])

    deg = 2
    def poly(xy, *coeffs):
        x, y = xy
        k = 0
        s = 0 
        for i in range(deg + 1):
            for j in range(deg + 1):
                s += coeffs[k] * np.power(x, i) * np.power(y, j)
                k += 1
        return s

    i0, j0 = minind 
    if i0 > 0 and i0 < h - 1 and j0 > 0 and j0 < w - 1:
        dir_ = get_dir_minmax(losses, minind, optm='min')
        # print(losses[i0-1, i0)
        breakpoint()
            
        # xx = np.array([steps[i0 - 1], steps[i0], steps[i0 + 1]])
        # yy = np.array([alphas[j0 - 1], alphas[j0], alphas[j0 + 1]])
        # xs, ys = np.meshgrid(xx, yy)
        # aux = np.array([-1, 0, +1])
        # ii, jj = np.meshgrid(aux, aux)
        # zflat = losses[ii, jj].flatten()
        # p0 = [1] * np.power(deg + 1, 2)
        # popt, pcov = curve_fit(poly, (xs.flatten(), ys.flatten()), zflat, p0)
        # breakpoint()
    

    return minind

##########################################################
def estimate_min(losses, smpvals, outdir):
    # TODO: Try to walk in the direction of min increase

    ind0 = (7, 1)
    lr0 = 10 
    predind, pathinds = gradient_descent(losses, ind0, lr0)
    # print(len(pathinds))
    steppred = smpvals['step'][predind[0]]
    alphapred = smpvals['alpha'][predind[1]]
    print('PRED: step, alpha, ind, loss:', steppred, alphapred, predind,
            losses[predind[0], predind[1]])
    breakpoint()
    
    return predind

##########################################################
def plot_wireframe3d(xs, ys, zs, outpath):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    # xx2, yy2 = np.meshgrid(alphas, times)
    
    # zz2 = fitfun((xx2, yy2), *popt)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_wireframe(xx2, yy2, data-zz2)
    # plt.savefig('/tmp/' + feat + '.png'); plt.close()
    

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_wireframe(xs, ys, zs)
    ax.set_xlabel('alpha')
    ax.set_ylabel('step')
    plt.savefig(outpath); plt.close()

##########################################################
def plot_scatter3d(x_data, y_data, z_data, outpath):
    fig, axes = plt.subplots(subplot_kw={"projection": "3d"})
    axes.scatter(x_data, y_data, z_data)
    axes.set_xlabel('alpha')
    axes.set_ylabel('step')
    # axes.set_zlabel(feat)
    plt.savefig(outpath); plt.close('all')

##########################################################
def fit_polynomials(deg, xs, ys, data, feats, outdir):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    xflat = xs.flatten()
    yflat = ys.flatten()
    p0 = [1] * np.power(deg + 1, 2)

    def poly(xy, *coeffs):
        x, y = xy
        k = 0
        s = 0 
        for i in range(deg + 1):
            for j in range(deg + 1):
                s += coeffs[k] * np.power(x, i) * np.power(y, j)
                k += 1
        return s

    for i, feat in enumerate(feats):
        zs = data[i, :, :]
        zflat = zs.flatten()
        popt, pcov = curve_fit(poly, (xflat, yflat), zflat, p0)
        zspred = poly((xs, ys), *popt)
        
        # plot_wireframe3d(xs, ys, zs, pjoin(outdir, feat + '_orig_wframe.png'))
        plot_scatter3d(xs, ys, zs, pjoin(outdir, feat + '_1orig_wframe.png'))
        plot_wireframe3d(xs, ys, zspred, pjoin(outdir, feat + '_2pred_wframe.png'))
        plot_wireframe3d(xs, ys, zs - zspred, pjoin(outdir, feat + '_3diff_wframe.png'))

##########################################################
def get_ranges(df, feats):
    franges = []
    for feat in feats:
        uvals = np.unique(df[feat])
        franges.append([np.min(uvals), np.max(uvals)])
    return franges

##########################################################
def interpolate_one_axis(data, smpvals):
    """Short description """
    info(inspect.stack()[0][3] + '()')
    dd = data[0, :, :]
    h, w  = dd.shape
    vtgt = 400
    diff = dd - vtgt
    inds = np.where(diff == np.min(np.abs(diff)))
    ic, jc = [inds[0][0], inds[1][0]] # Get just the first one
    
    print(ic, jc)

    if ic == 0 or ic == h - 1: #TODO: fix this later
        s2 = smpvals['step'][ic]
    else:
        i1 = ic - 1; i3 = ic + 1
        s1 = smpvals['step'][i1]; s3 = smpvals['step'][i3];
        v1 = dd[i1, jc]; v2 = vtgt; v3 = dd[i3, jc]
        s2 = (v2-v1) / (v3-v1) * (s3-s1) + s1

    if jc == 0 or jc == w - 1: #TODO: fix this later
        a2 = smpvals['alpha'][jc]
    else:
        j1 = jc - 1; j3 = jc + 1
        a1 = smpvals['alpha'][j1]; a3 = smpvals['alpha'][j3];
        v1 = dd[i1, jc]; v2 = vtgt; v3 = dd[i3, jc]
        a2 = (v2-v1) / (v3-v1) * (a3-a1) + a1

##########################################################
def main(outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')

    feats = ['vcount', 'recipr', 'k']

    respath = './data/results.csv'
    df = pd.read_csv(respath)

    franges = get_ranges(df, feats)
    # plot_features_across_time(df, feats, outdir)
    # print_features_latex(df, feats)
    # data, smpvals = get_average_values(df, feats, 1);  interpolate_one_axis(data, smpvals)

    data, smpvals = get_average_values(df, feats, 100)
    losses = get_loss(data, tgt=[300, .25, 3], outdir=outdir)
    minidx = get_min(losses, smpvals, outdir)
    lossmin = estimate_min(losses, smpvals, outdir)
    breakpoint()
    
    return
    
    # data, smpvals = get_average_values(df, feats, 300)
    # data = data[:, 1:, :]
    # smpvals['step'] =  smpvals['step'][1:]
    xs, ys = np.meshgrid(smpvals['alpha'], smpvals['step'])

    os.makedirs(pjoin(outdir, 'poly2'), exist_ok=True)
    fit_polynomials(2, xs, ys, data, feats, pjoin(outdir, 'poly2'))
    os.makedirs(pjoin(outdir, 'poly3'), exist_ok=True)
    fit_polynomials(3, xs, ys, data, feats, pjoin(outdir, 'poly3'))

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
