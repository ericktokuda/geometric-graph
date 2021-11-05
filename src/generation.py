#!/usr/bin/env python3
"""Generation of a geometric graph
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
from sklearn import neighbors
import scipy
from scipy.spatial.distance import cdist

##########################################################
def main(outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')

    n = 3
    coords = np.random.rand(n, 2)
    # coords = np.array([[1,1], [4,4], [2,2]])

    a = 0.5
    b = 1
    v0 = np.random.randint(n)
    l = 3
    dists = cdist(coords, coords)
    aux = b * np.exp(- a * dists)
    probs = aux / np.sum(aux, axis=1)[:, np.newaxis]

    randvals = np.random.rand(l)
    cumsums = np.zeros((n, n), dtype=float)
    inds = np.zeros((n, n), dtype=int)

    # Set up ranges for random sampling
    ranges = []
    for i in range(n):
        pr = probs[i, :]
        inds[i, :] = np.argsort(-pr) # We check the nodes with largest probs first
        cumsums[i, :] = np.cumsum(pr[inds[i, :]])

    walk = np.zeros(n + 1, dtype=int)
    walk[0] = v0
    for i in range(l):
        vcur = walk[i]
        # print(vcur)
        sampled = randvals[i]
        bin0 = 0
        for j in range(1, n): # First is self (distance 0)
            bin1 = cumsums[vcur, j]
            # print(bin0, bin1)
            if (sampled > bin0) and (sampled <= bin1):
                walk[i+1] = inds[vcur, j]
                break
            bin0 = bin1
        assert j <= n

    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    ax.scatter(coords[:, 0], coords[:, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i in range(len(coords)):
        x, y = coords[i, :];
        label = str(i)
        ax.annotate(label,  xy=(x, y), color='k', weight='heavy',
                    horizontalalignment='center',
                    verticalalignment='center', zorder=11)

    outpath = '/tmp/foo.png'
    plt.savefig(outpath)
    print(walk)

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
