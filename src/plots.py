#!/usr/bin/env python3
"""Plots """

import argparse
import time, datetime
import os
from os.path import join as pjoin
import inspect

import sys
import random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from myutils import info, create_readme
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit, leastsq
from sklearn.metrics import r2_score
from sympy import symbols

##########################################################
W = 1200; H = 960

##########################################################
def poly(xy, deg, *coeffs):
    x, y = xy
    k = 0
    s = 0
    for i in range(deg + 1):
        for j in range(deg + 1):
            s += coeffs[k] * np.power(x, i) * np.power(y, j)
            k += 1
    return s

##########################################################
def poly2(xy, *coeffs): return poly(xy, 2, *coeffs)
def poly3(xy, *coeffs): return poly(xy, 3, *coeffs)
def fun3(x, y, *coeffs): return poly((x, y), 3, *coeffs) # Sympy-friendly format
POLYS = {2: poly2, 3: poly3}

##########################################################
def get_dir_minmax(losses, ind0, optm='max'):
    """Get the neighbouring pixel which corresponds to the minimum/maximum gradient
    magnitude"""
    ind0 = np.array(ind0)
    curloss = losses[ind0[0], ind0[1]]
    
    # x0, y0 = ind
    dirs_ = np.array([
        [+ 1,   0 ], [  0, +1 ],
        [- 1,   0 ], [  0, -1 ],
        [- 1,  -1 ], [ +1, +1 ],
        ])

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
def get_matrix_dir(losses, ind, method='greedy'):
    """Choose the 8-connected direction based on neighbourhood given that we are
    trying to minimize the losses"""
    if method == 'greedy':
        return get_dir_minmax(losses, ind, 'min')
    elif method == 'opposmax':
        return - get_dir_minmax(losses, ind, 'max')

##########################################################
def get_matrix_min_inds(losses):
    """Get the index of minimum elements"""
    info(inspect.stack()[0][3] + '()')
    n, w, h = losses.shape
    mininds = []
    
    for i in range(n):
        loss = losses[i, :, :]
        minind = np.unravel_index(np.argmin(loss), loss.shape)
        mininds.append(list(minind))
    return mininds

##########################################################
def gradient_descent(fsym, dfdxsym, dfdysym, p0, lr):
    info(inspect.stack()[0][3] + '()')
    errthresh = 1e-3 # Error threshold
    maxsteps = 1000
    curerr = 99999
    visitted = [] # Store the descent path
    x, y = symbols('x y')
    p = p0
    pace = [1, 1]

    for step in range(maxsteps):
        visitted.append(p)
        curerr = fsym.subs([(x, p[0]), (y, p[1])])
        if (curerr < errthresh) or (np.linalg.norm(pace) < errthresh): break
        lr *= .95
        dfdx = dfdxsym.subs([(x, p[0]), (y, p[1])])
        dfdy = dfdysym.subs([(x, p[0]), (y, p[1])])
        grad = np.array([dfdx, dfdy]).astype(float)
        print(dfdx, dfdy)
        grad = grad / np.linalg.norm(grad)
        # print(p, np.linalg.norm(pace), curerr)
        pace = - lr * grad
        p = p + pace

    return visitted

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
    """Get average values across realizations of the features @feats,
    sampled every @spl steps."""
    data = []
    for feat in feats:
        xs = np.unique(df.step)[::spl]
        # xs = np.unique(df.step)[::spl] + 1
        # if xs[-1] >= np.max(df.step): xs = xs[:-1]
        d = []
        for a in sorted(np.unique(df.alpha)):
            df2 = df.loc[df.alpha == a]
            ys = df2.groupby(['step'])[feat].mean()[xs]
            d.append(ys)
        data.append(np.array(d))
    smpvals = {'step': xs, 'alpha' : np.unique(df.alpha)}
    return np.array(data), smpvals

##########################################################
def get_loss(data, tgt):
    """Loss is defined as the euclidean difference between the @tgt and each row
    in @data. We normalize the @data column-wise and apply the same normalization
    to @tgt."""
    # info(inspect.stack()[0][3] + '()')
    loss = np.zeros((data.shape[1], data.shape[2]))
    for i in range(data.shape[0]):
        vmin, vmax = np.min(data[i, :, :]), np.max(data[i, :, :])
        d = (data[i, :, :] - vmin) / (vmax - vmin)
        t = (tgt[i] - vmin) / (vmax - vmin)
        diff = d - t
        loss += np.power(diff, 2)
    return loss

##########################################################
def plot_loss_heatmap(loss, outpath):
    scaler = MinMaxScaler((0, 255))
    normloss = scaler.fit_transform(loss).astype(int)
    plt.imshow(normloss)
    plt.colorbar()
    plt.savefig(outpath); plt.close()

##########################################################
def plot_wireframe3d(xs, ys, zs, outpath):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_wireframe(xs, ys, zs)
    ax.set_xlabel('alpha')
    ax.set_ylabel('step')
    ax.invert_xaxis()
    plt.savefig(outpath); plt.close()

##########################################################
def plot_contours(xs, ys, zs, outpath=''):
    fig, ax = plt.subplots()
    cont = ax.contour(xs, ys, zs)
    ax.set_xlabel('alpha')
    ax.set_ylabel('step')
    ax.clabel(cont)
    if outpath:
        plt.savefig(outpath); plt.close()
    else:
        return fig, ax

##########################################################
def plot_scatter3d(x_data, y_data, z_data, outpath):
    fig, axes = plt.subplots(subplot_kw={"projection": "3d"})
    axes.scatter(x_data, y_data, z_data)
    axes.set_xlabel('alpha')
    axes.set_ylabel('step')
    # axes.set_zlabel(feat)
    plt.savefig(outpath); plt.close('all')

##########################################################
def fit_polynomials(deg, xs, ys, data):
    """Fit a polynomial of @deg degree in @data"""
    info(inspect.stack()[0][3] + '()')
    xflat = xs.flatten()
    yflat = ys.flatten()
    p0 = [1] * np.power(deg + 1, 2)

    if deg == 2: fun = poly2
    elif deg == 3: fun = poly3

    popts = []
    for i in range(data.shape[0]):
        zs = data[i, :, :]
        zflat = zs.flatten()
        popt, pcov = curve_fit(fun, (xflat, yflat), zflat, p0)
        zspred = fun((xs, ys), *popt)
        popts.append(popt)
    return fun, popts

##########################################################
def fit_polynomial(xs, ys, zs, deg):
    """Fit a polynomial of degree @deg"""
    info(inspect.stack()[0][3] + '()')
    xflat = xs.flatten()
    yflat = ys.flatten()
    p0 = [1] * np.power(deg + 1, 2)

    fun = POLYS[deg]

    zflat = zs.flatten()
    popt, pcov = curve_fit(fun, (xflat, yflat), zflat, p0)
    zspred = fun((xs, ys), *popt)
    # plot_scatter3d(xs, ys, zs, pjoin(outdir, feat + '_1orig_scatter.png'))
    # plot_wireframe3d(xs, ys, zspred, pjoin(outdir, feat + '_2pred_wframe.png'))
    # plot_wireframe3d(xs, ys, zs - zspred, pjoin(outdir, feat + '_3diff_wframe.png'))

    avgerr = np.mean(np.abs(zflat - zspred.flatten()))
    # r2 = r2_score(zflat, zspred.flatten())
    
    return fun, popt, avgerr

##########################################################
def get_ranges(df, feats):
    franges = []
    for feat in feats:
        uvals = np.unique(df[feat])
        franges.append([np.min(uvals), np.max(uvals)])
    return np.array(franges)

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
def get_losses(data, tgts):
    """Plot loss of the original points, restricted to the sampled grid"""
    losses = {}
    for tt, tgt in enumerate(tgts):
        k = '{:.02f}_{:.02f}_{:.02f}'.format(*tgt)
        losses[k] = get_loss(data, tgt)
    return losses

##########################################################
def plot_losses(losses, xs, ys, outdir):
    os.makedirs(outdir, exist_ok=True)
    
    for tgt, v in losses.items():
        # plot_loss_heatmap(loss, pjoin(outdir, 'loss.png'))
        outpath = pjoin(outdir, '{}.png'.format(tgt))
        plot_wireframe3d(xs, ys, v, outpath)
        outpath = pjoin(outdir, 'contour_{}.png'.format(tgt))
        # plot_contours(xs, ys, v, outpath)

##########################################################
def generate_targets(franges, samplesz):
    """Generate uniform random targets (3-uples)"""
    info(inspect.stack()[0][3] + '()')
    rnd = np.random.rand(samplesz, 3)
    
    for i in range(franges.shape[0]):
        vmin, vmax = franges[i, :]
        rnd[:, i] = rnd[:, i] * (vmax - vmin) + vmin
    return rnd

##########################################################
def eval_fun(fun, popts, xs, ys):
    """Evaluate function @fun, with parameters @popts at (@xs,@ys)"""
    n, w, h = len(popts), xs.shape[0], xs.shape[1]
    fitted = np.zeros((n, w, h), dtype=float)
    for i in range(fitted.shape[0]):
        for j in range(fitted.shape[1]):
            for k in range(fitted.shape[2]):
                fitted[i, j, k] = fun((xs[j, k], ys[j, k]), *(popts[i]))
    return fitted

##########################################################
def get_gradient_poly3():
    x, y, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p = symbols(
            'x y a b c d e f g h i j k l m n o p')
    u = fun3(x, y, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
    grad = [u.diff(x), u.diff(y)]
    return grad

##########################################################
def lossfun(x, y, t1, t2, t3, *coeffs):
    coeffs1 = coeffs[0 : 16]
    coeffs2 = coeffs[16: 32]
    coeffs3 = coeffs[32: 48]
    loss = np.power(poly((x, y), 3, *coeffs1) - t1, 2) + \
        np.power(poly((x, y), 3, *coeffs2) - t2, 2) + \
        np.power(poly((x, y), 3, *coeffs3) - t3, 2)
    return loss

##########################################################
def main(outdir):
    info(inspect.stack()[0][3] + '()')

    seed = 0
    random.seed(seed); np.random.seed(seed)
    feats = ['vcount', 'recipr', 'k']
    respath = './data/results.csv'
    df = pd.read_csv(respath)

    franges = get_ranges(df, feats) # Feature value ranges
    # plot_features_across_time(df, feats, outdir)
    # print_features_latex(df, feats)

    tgtsorig = generate_targets(franges, samplesz=1)
    
    data, smpvals = get_average_values(df, feats, 100)

    # # Alpha and Time grid with shapes: (nalphas, nsteps)
    tgrid, agrid = np.meshgrid(smpvals['step'], smpvals['alpha'])
    
    lossesorig = get_losses(data, tgtsorig)
    plot_losses(lossesorig, agrid, tgrid, pjoin(outdir, 'orig'))

    # # Normalize data and tgt
    tgts = np.zeros(tgtsorig.shape)
    for i in range(3):
        vmin, vmax = np.min(data[i, :, :]), np.max(data[i, :, :])
        data[i, :, :] = (data[i, :, :] - vmin) / (vmax - vmin)
        tgts[:, i] = (tgtsorig[:, i] - vmin) / (vmax - vmin)

    
    mininds = {}
    minvals = {}
    for k in lossesorig:
        loss = lossesorig[k]
        z = np.unravel_index(np.argmin(loss), loss.shape)
        mininds[k] = [smpvals['alpha'][z[0]], smpvals['step'][z[1]]]
        minvals[k] = np.min(loss)
    
    print('Tgt:', tgtsorig[0])
    print('Min params:', list(mininds.values()))
    print('Min loss:', list(minvals.values()))

    poly, popts = fit_polynomials(3, agrid, tgrid, data)

    xx, yy = agrid, tgrid # Could be any other sampling
    fitted = eval_fun(poly, popts, xx, yy)
    lossesfitted = get_losses(fitted, tgts) # Just used to plot
    plot_losses(lossesfitted, agrid, tgrid, pjoin(outdir, 'fitted'))
    
    x, y, t1, t2, t3 = symbols('x y t1 t2 t3')
    a1, b1, c1, d1, e1, f1, g1, h1, i1, j1, k1, l1, m1, n1, o1, p1 = symbols(
        'a1 b1 c1 d1 e1 f1 g1 h1 i1 j1 k1 l1 m1 n1 o1 p1')
    a2, b2, c2, d2, e2, f2, g2, h2, i2, j2, k2, l2, m2, n2, o2, p2 = symbols(
        'a2 b2 c2 d2 e2 f2 g2 h2 i2 j2 k2 l2 m2 n2 o2 p2')
    a3, b3, c3, d3, e3, f3, g3, h3, i3, j3, k3, l3, m3, n3, o3, p3 = symbols(
        'a3 b3 c3 d3 e3 f3 g3 h3 i3 j3 k3 l3 m3 n3 o3 p3')

    f = lossfun(x, y, t1, t2, t3,
                a1, b1, c1, d1, e1, f1, g1, h1, i1, j1, k1, l1, m1, n1, o1, p1,
                a2, b2, c2, d2, e2, f2, g2, h2, i2, j2, k2, l2, m2, n2, o2, p2,
                a3, b3, c3, d3, e3, f3, g3, h3, i3, j3, k3, l3, m3, n3, o3, p3)

    dfdx, dfdy = f.diff(x), f.diff(y)
    vars1 = [a1, b1, c1, d1, e1, f1, g1, h1, i1, j1, k1, l1, m1, n1, o1, p1]
    vars2 = [a2, b2, c2, d2, e2, f2, g2, h2, i2, j2, k2, l2, m2, n2, o2, p2]
    vars3 = [a3, b3, c3, d3, e3, f3, g3, h3, i3, j3, k3, l3, m3, n3, o3, p3]
    vars = vars1 + vars2 + vars3
    repl = [(v, p) for v, p in zip(vars, np.array(popts).flatten())]

    f2, dfdx2, dfdy2  = f.subs(repl), dfdx.subs(repl), dfdy.subs(repl)

    lr0 = 20
    p0 = np.array([20, 2000]).astype(float)
    for tgt in tgts:
        f3    =    f2.subs([(t1, tgt[0]), (t2, tgt[1]), (t3, tgt[2])])
        dfdx3 = dfdx2.subs([(t1, tgt[0]), (t2, tgt[1]), (t3, tgt[2])])
        dfdy3 = dfdy2.subs([(t1, tgt[0]), (t2, tgt[1]), (t3, tgt[2])])
        visinds = gradient_descent(f3, dfdx3, dfdy3, p0, lr0)
        for i, v in enumerate(visinds):
            k = '{:.02f}_{:.02f}_{:.02f}'.format(*tgt)
            fig, ax = plot_contours(agrid, tgrid, lossesfitted[k])
            ax.scatter(v[0], v[1])
            plt.savefig(pjoin('/tmp/{:02d}.png'.format(i)))
            plt.close()

        # breakpoint()
        
        # print(visinds[-1])
        
    # print(pmin)

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
