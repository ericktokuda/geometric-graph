#!/usr/bin/env python3
"""Plots
"""

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

    respath = './data/resultsall.csv'
    df = pd.read_csv(respath)

    W = 640; H = 480

    for feat in ['vcount', 'acount', 'recipr', 'k']:
        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        for a in sorted(np.unique(df.alpha)):
            df2 = df.loc[df.alpha == a]
            ax.plot(df2.step, df2[feat], alpha=.5, label=a)
            outpath = pjoin(outdir, feat + '.png')
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
