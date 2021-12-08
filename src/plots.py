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
def get_dir_max(losses, ind0):
    """Get the neighbouring pixel which corresponds to the largest gradient
    magnitude"""
    ind0 = np.array(ind0)
    curloss = losses[ind0[0], ind0[1]]
    
    # x0, y0 = ind
    dirs_ = np.array([
        [+ 1,   0 ], [  0, + 1 ],
        [- 1,   0 ], [  0, - 1 ], ])

    maxdiff = 0
    dirmax = ind0
    h, w = losses.shape

    for dir_ in dirs_:
        i, j = np.array(ind0 + dir_)
        if (i < 0) or (i >= h) or (j < 0) or (j >= w): continue
        curdiff = losses[i, j] - curloss
        if curdiff > maxdiff:
            dirmax = dir_
            maxdiff = curdiff
    return dirmax

##########################################################
def gradient_descent(losses, ind0, lr0):
    errthresh = 1e-3 # Error threshold
    maxsteps = 1000
    curerr = 99999
    step = 0
    lr = lr0
    losstgt = 0 # Target loss

    h, w = losses.shape

    ind = ind0
    while (step < maxsteps) and (curerr > errthresh):
        dirmax =  get_dir_max(losses, ind0)
        ind = ind - lr * dirmax
        ind = np.array([int(np.round(ind[0])), int(np.round(ind[1]))])

        if ind[0] < 0: ind[0] = 0 # Clip values
        elif ind[0] > h: ind[0] = h - 1
        if ind[1] < 0: ind[1] = 0
        elif ind[1] >= w: ind[1] = w - 1

        curloss = losses[ind[0], ind[1]]
        curerr = np.abs(losstgt - curloss)
        step += 1
    return ind

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
def find_minimum(losses, smpvals, outdir):
    """Find the minimum (interpolate)"""
    info(inspect.stack()[0][3] + '()')
    minind = np.unravel_index(np.argmin(losses), losses.shape)
    steppred = smpvals['step'][minind[0]]
    alphapred = smpvals['alpha'][minind[1]]
    print('Min step, pred:', steppred, alphapred)
    print('Minind:{}'.format(minind))

    ind0 = (5, 5)
    lr0 = 5
    predind = gradient_descent(losses, ind0, lr0)

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
    plt.savefig(outpath); plt.close()

##########################################################
def plot_scatter3d(x_data, y_data, z_data):
    fig, axes = plt.subplots(subplot_kw={"projection": "3d"})
    axes.scatter(x_data, y_data, z_data)
    axes.set_title('Scatter Plot (click-drag with mouse)')
    axes.set_xlabel('alpha')
    axes.set_ylabel('step')
    axes.set_zlabel(feat)
    plt.savefig(pjoin(outdir, feat + '.png')); plt.close('all')

##########################################################

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
        
        plot_wireframe3d(xs, ys, zs, pjoin(outdir, feat + '_orig_wframe.png'))
        plot_wireframe3d(xs, ys, zspred, pjoin(outdir, feat + '_pred_wframe.png'))
        plot_wireframe3d(xs, ys, zs - zspred, pjoin(outdir, feat + '_diff_wframe.png'))

##########################################################
def main(outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')

    feats = ['vcount', 'recipr', 'k']

    respath = './data/results.csv'
    df = pd.read_csv(respath)

    # plot_features_across_time(df, feats, outdir)
    # print_features_latex(df, feats)
    data, smpvals = get_average_values(df, feats, 300)
    # losses = get_loss(data, tgt=[400, .25, 3], outdir=outdir)
    # lossmin = find_minimum(losses, smpvals, outdir)
    xs, ys = np.meshgrid(smpvals['alpha'], smpvals['step'])

    # fit_polynomials(2, xs, ys, data, feats, outdir)
    fit_polynomials(3, xs, ys, data, feats, outdir)


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
