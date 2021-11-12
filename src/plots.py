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
    df = pd.read_csv(respath)

    W = 1200; H = 960
    spl = 10 # Plot every spl points

    for feat in ['vcount', 'acount', 'recipr', 'k']:
        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)

        for a in sorted(np.unique(df.alpha)):
            df2 = df.loc[df.alpha == a]
            xs = np.unique(df2.step.values)[::spl]
            ys = df2.groupby(['step'])[feat].mean()[::spl]
            yerr = df2.groupby(['step'])[feat].std()[::spl]
            ax.errorbar(xs, ys, yerr, alpha=.5, label=a)
            ax.set_ylabel(feat.capitalize())
            ax.set_xlabel('Time')
            outpath = pjoin(outdir, feat + '.pdf')
        plt.legend()
        plt.savefig(outpath); plt.close()
    info('For Aiur!')


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
