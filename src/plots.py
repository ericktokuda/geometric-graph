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
from myutils import info, create_readme
import pandas as pd

##########################################################
def main(outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')

    respath = './data/results.csv'
    # respath = '/home/frodo/results/geometricgraph/20211108-geom/uniform/results.csv'
    df = pd.read_csv(respath)
    steps = np.unique(df.step)
    alphas = np.unique(df.alpha)

    W = 1200; H = 960
    spl = 10 # Plot every spl points

    # Plot evolution in time of each feature
    for feat in ['vcount', 'acount', 'recipr', 'k']:
        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        xs = steps[::spl]
        for a in sorted(alphas):
            df2 = df.loc[df.alpha == a]
            ys = df2.groupby(['step'])[feat].mean()[::spl]
            yerr = df2.groupby(['step'])[feat].std()[::spl]
            # ax.plot(xs, np.power(ys, 3), alpha=.5, label=a)
            ax.errorbar(xs, ys, yerr, alpha=.5, label=a)
            ax.set_ylabel(feat.capitalize())
            ax.set_xlabel('Time')
        outpath = pjoin(outdir, feat + '.pdf')
        plt.legend()
        plt.savefig(outpath); plt.close()

    # Print latex
    spl = 300
    for feat in ['vcount', 'recipr', 'k']:
        data = []
        xs = steps[::spl]
        for a in sorted(alphas):
            df2 = df.loc[df.alpha == a]
            ys = df2.groupby(['step'])[feat].mean()[::spl]
            # yerr = df2.groupby(['step'])[feat].std()[::spl]
            data.append(ys)
        data = np.array(data).T
        df3 = pd.DataFrame(data, columns=[str(a) for a in alphas])
        # print(df3.to_latex())

    # Plot image req. by luciano
    spl = 1 # Plot every spl points
    data = []
    for feat in ['vcount', 'recipr', 'k']:
        xs = steps[::spl]
        d = []
        for a in sorted(alphas):
            df2 = df.loc[df.alpha == a]
            ys = df2.groupby(['step'])[feat].mean()[::spl]
            # yerr = df2.groupby(['step'])[feat].std()[::spl]
            d.append(ys)
        data.append(np.array(d).T)
    data = np.array(data)

    tgt = [400, .25, 3] # nvert, recipr, k

    loss = np.zeros((data.shape[1], data.shape[2]))
    for i in range(data.shape[0]):
        loss += np.power(data[i, :, :] - tgt[i], 2)
    ind = np.unravel_index(np.argmin(loss, axis=None), loss.shape)

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
