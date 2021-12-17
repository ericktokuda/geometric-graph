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
from sklearn.metrics import r2_score
import sympy

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
def gradient_descent_old(losses, ind0, lr0):
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
def gradient_descent(fsym, dfdxsym, dfdysym, p0, lr):
    info(inspect.stack()[0][3] + '()')
    errthresh = 1e-3 # Error threshold
    maxsteps = 1000
    curerr = 99999
    visitted = [] # Store the descent path
    x, y = sympy.symbols('x y')
    p = p0
    pace = [1, 1]

    for step in range(maxsteps):
        curerr = fsym.subs([(x, p[0]), (y, p[1])])
        if (curerr < errthresh) or (np.linalg.norm(pace) < errthresh): break
        lr *= .9
        dfdx = dfdxsym.subs([(x, p[0]), (y, p[1])])
        dfdy = dfdysym.subs([(x, p[0]), (y, p[1])])
        grad = np.array([dfdx, dfdy]).astype(float)
        grad = grad / np.linalg.norm(grad)
        print(p, dfdx, dfdy, curerr)
        pace = - lr * grad
        p = p + pace

    return p

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
def get_loss(data, tgt):
    """Loss is defined as the euclidean difference between the @tgt and each row in
    @data. We normalize the @data column-wise and apply the same normalization to
    @tgt."""
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
        # TODO: choose a fractional step size to walk in opposite direction of dir_
        # ii, jj = np.meshgrid([i0-1, i0, i0+1], [j0-1, j0, j0+1])
        # print(losses[ii, jj] - losses[i0, j0])
        # print(dir_)
        # print(losses[ii, jj])
        # print(losses[ii, jj].astype(int))
            
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
def estimate_min(tgt, data, smpvals, outdir):

    xs, ys = np.meshgrid(smpvals['alpha'], smpvals['step'])
    popts = []
    m = data.shape[0]
    for featidx in range(m):
        zs = data[featidx, :, :]
        f, popt, err = fit_polynomial(xs, ys, zs, 2)
        popts.append(popt)

    x0, y0 = 100, 22
    costs2 = [ np.power(tgt[i] - f((x0, y0), *(popts[i])), 2) for i in range(m) ]
    # cost2 = 0
    # for i in range(m): cost2 += np.power(tgt[i] - f((x0, y0), *(popts[i])), 2)
    cost = np.sqrt(np.sum(costs2))
    print(cost)
    breakpoint()
    
    ind0 = (7, 1)
    lr0 = 100 
    predind, pathinds = gradient_descent(losses, ind0, lr0)
    # print(len(pathinds))
    steppred = smpvals['step'][predind[0]]
    alphapred = smpvals['alpha'][predind[1]]
    print('PRED: step, alpha, ind, loss:', steppred, alphapred, predind,
            losses[predind[0], predind[1]])
    
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
def fit_polynomials(deg, xs, ys, data):
    """Short description """
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
def plot_losses(data, xs, ys, tgts, outdir):
    """Plot loss of the original points, restricted to the sampled grid"""
    info(inspect.stack()[0][3] + '()')

    # outdir = pjoin(outdir, 'orig')
    os.makedirs(outdir, exist_ok=True)

    for tt, tgt in enumerate(tgts):
        losses = get_loss(data, tgt)
        # plot_loss_heatmap(loss, pjoin(outdir, 'loss.png'))
        outpath = pjoin(outdir, '{:.02f}_{:.02f}_{:.02f}.png'.format(*tgt))
        plot_wireframe3d(xs, ys, losses, outpath)

##########################################################
def plot_losses_fitted(data, xs, ys, tgts, outdir):
    """Plot loss considering the 3rd order polynomial surface fitted to @data and
    the target"""
    info(inspect.stack()[0][3] + '()')

    outdir = pjoin(outdir, 'fitted')
    os.makedirs(outdir, exist_ok=True)

    for tt, tgt in enumerate(tgts):
        outpath = pjoin(outdir, '{:.02f}_{:.02f}_{:.02f}.png'.format(*tgt))

        fun, popts = fit_polynomials(3, xs, ys, data)
        fitted = np.zeros(data.shape, dtype=float)
        for i in range(fitted.shape[0]):
            for j in range(fitted.shape[1]):
                for k in range(fitted.shape[2]):
                    fitted[i, j, k] = fun((xs[j, k], ys[j, k]), *(popts[i]))

        loss = get_loss(fitted, tgt)
        plot_wireframe3d(xs, ys, loss, outpath)

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
def sample_fun(fun, xs, ys, popts):
    info(inspect.stack()[0][3] + '()')
    n, w, h = len(popts), xs.shape[0], xs.shape[1]
    fitted = np.zeros((n, w, h), dtype=float)
    for i in range(fitted.shape[0]):
        for j in range(fitted.shape[1]):
            for k in range(fitted.shape[2]):
                fitted[i, j, k] = fun((xs[j, k], ys[j, k]), *(popts[i]))
    return fitted

##########################################################
def get_gradient_poly3():
    x, y, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p = sympy.symbols(
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

    feats = ['vcount', 'recipr', 'k']
    respath = './data/results.csv'
    df = pd.read_csv(respath)

    franges = get_ranges(df, feats)
    # plot_features_across_time(df, feats, outdir)
    # print_features_latex(df, feats)

    tgts = generate_targets(franges, samplesz=5)
    data, smpvals = get_average_values(df, feats, 100)
    xs, ys = np.meshgrid(smpvals['alpha'], smpvals['step'])

    # plot_losses(data, xs, ys, tgts, pjoin(outdir, 'orig'))

    # Normalize data and tgt
    for i in range(3):
        d = data[i, :, :]
        vmin, vmax = np.min(d), np.max(d)
        data[i, :, :] = (d - vmin) / (vmax - vmin)
        tgts[i] = (tgts[i] - vmin) / (vmax - vmin)

    poly, popts = fit_polynomials(3, xs, ys, data)
    fitted = sample_fun(poly, xs, ys, popts) # sampled points is arbitrary in the range
    # plot_losses(fitted, xs, ys, tgts, pjoin(outdir, 'fitted'))
    # dir = get_dir_minmax(fitted[0, :, :], [5, 5], optm='max')

    x, y, t1, t2, t3 = sympy.symbols('x y t1 t2 t3')
    a1, b1, c1, d1, e1, f1, g1, h1, i1, j1, k1, l1, m1, n1, o1, p1 = sympy.symbols(
        'a1 b1 c1 d1 e1 f1 g1 h1 i1 j1 k1 l1 m1 n1 o1 p1')
    a2, b2, c2, d2, e2, f2, g2, h2, i2, j2, k2, l2, m2, n2, o2, p2 = sympy.symbols(
        'a2 b2 c2 d2 e2 f2 g2 h2 i2 j2 k2 l2 m2 n2 o2 p2')
    a3, b3, c3, d3, e3, f3, g3, h3, i3, j3, k3, l3, m3, n3, o3, p3 = sympy.symbols(
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

    f2    =    f.subs(repl)
    dfdx2 = dfdx.subs(repl)
    dfdy2 = dfdy.subs(repl)

    lr0 = 10
    p0 = np.array([500, 20]).astype(float)
    p0 = (p0 - vmin) / (vmax - vmin)
    for tgt in tgts:
        loss = get_loss(fitted, tgt)
        f3    =    f2.subs([(t1, tgt[0]), (t2, tgt[1]), (t3, tgt[2])])
        dfdx3 = dfdx2.subs([(t1, tgt[0]), (t2, tgt[1]), (t3, tgt[2])])
        dfdy3 = dfdy2.subs([(t1, tgt[0]), (t2, tgt[1]), (t3, tgt[2])])
        pmin, visitted = gradient_descent(f3, dfdx3, dfdy3, p0, lr0)
    print(pmin)

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
