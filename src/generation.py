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
import random
import pandas as pd

##########################################################
def plot_walk1(walk, coords, outdir):
    """Plot a single figure with the walk steps labeled close to the vertices """
    info(inspect.stack()[0][3] + '()')
    W = 640; H = 480
    fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
    ax.scatter(coords[:, 0], coords[:, 1])
    margin = .05
    ax.set_xlim(0 - margin, 1 + margin)
    ax.set_ylim(0 - margin, 1 + margin)

    for i in range(len(coords)):
        x, y = coords[i, :] + .02
        label = str(i)
        ax.annotate(label,  xy=(x, y), color='k', weight='heavy',
                    horizontalalignment='center',
                    verticalalignment='center', zorder=11)

    for i in range(len(walk)):
        x, y = coords[walk[i], :] + np.array([np.random.rand() * 0.05, -.02])
        label = 'w{}'.format(i)
        ax.annotate(label,  xy=(x, y),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='red', alpha=0.5, size=10)

    outpath = pjoin(outdir, 'graph.png')
    plt.savefig(outpath)

##########################################################
def plot_walk2(walk, coords, outdir):
    """Plot a figure for each step (slow)"""
    info(inspect.stack()[0][3] + '()')
    W = 640; H = 480

    for i in range(1, len(walk)):
        fig, ax = plt.subplots(figsize=(W*.01, H*.01), dpi=100)
        ax.scatter(coords[:, 0], coords[:, 1], facecolors='none', edgecolors='k')
        margin = .05
        vprev = walk[i-1]
        vcur = walk[i]
        ax.scatter(coords[vprev, 0], coords[vprev, 1], facecolors='k', alpha=.3)
        ax.scatter(coords[vcur, 0], coords[vcur, 1], facecolors='k', alpha=1)

        ax.set_xlim(0 - margin, 1 + margin)
        ax.set_ylim(0 - margin, 1 + margin)

        outpath = pjoin(outdir, 'step{:03d}'.format(i))
        plt.savefig(outpath); plt.close()

##########################################################
def main(seed, npoints, a, outdir):
    info(inspect.stack()[0][3] + '()')

    random.seed(seed); np.random.seed(seed)

    coords = np.random.rand(npoints, 2)
    l = npoints * 10
    b = 1
    v0 = np.random.randint(npoints) # Choice of starting vertex

    aux = b * np.exp(- a * cdist(coords, coords)) # b*e^(a*d)
    probs = aux / np.sum(aux, axis=1)[:, np.newaxis]

    # Set up ranges for random sampling
    inds = np.zeros((npoints, npoints), dtype=int)
    cumsums = np.zeros((npoints, npoints), dtype=float)
    ranges = []
    for i in range(npoints):
        pr = probs[i, :]
        inds[i, :] = np.argsort(-pr) # We check the nodes with largest probs first
        cumsums[i, :] = np.cumsum(pr[inds[i, :]])

    # Walk
    adj = np.zeros((npoints, npoints), dtype=int)
    randvals = np.random.rand(l)
    walk = np.zeros(l + 1, dtype=int) # vertices after i steps in the walk
    paired = np.zeros(l + 1, dtype=int)
    ms = np.zeros(l + 1, dtype=int)
    ns = np.zeros(l + 1, dtype=int)
    walk[0] = v0; ns[0] = 1

    for i in range(l):
        vcur = walk[i] # walk[i] -> walk[i+1]
        sampled = randvals[i]
        bin0 = 0
        for j in range(1, npoints): # First is self (NOT allowing loops in the graph)
            bin1 = cumsums[vcur, j]
            if (sampled > bin0) and (sampled <= bin1):
                walk[i+1] = inds[vcur, j]

                newarc = True if (adj[walk[i], walk[i+1]] == 0) else False
                newvtx = True if (walk[i+1] in walk[:i+1]) else False
                symm = True if adj[walk[i+1], walk[i]] > 0 else False

                adj[walk[i], walk[i+1]] += 1

                ns[i+1] = ns[i] + 1 if newvtx else ns[i]
                ms[i+1] = ms[i] + 1 if newarc else ms[i]
                paired[i+1] = ms[i] + 2 if (newarc and symm) else ms[i]
                break
            bin0 = bin1
        assert j <= npoints

    rec = paired.astype(float) / ms
    rec[0] = 0 # Reciprocity when there is no arcs (avoid division by zero)
    ks = ms / ns

    data = {}
    data['alpha'] = [a] * (l+1)
    data['walk'] = walk
    data['vcount'] = ns
    data['acount'] = ms
    data['recipr'] = rec
    data['k'] = ks

    df = pd.DataFrame(data)
    csvpath = pjoin(outdir, 'results.csv')
    df.to_csv(csvpath, index=True, index_label='step')

    # Plot
    # plot_walk1(walk, coords, outdir)
    # plot_walk2(walk, coords, outdir)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npoints', default=40, type=int, help='Number of points')
    parser.add_argument('--alpha', default=20., type=float, help='Factor of the exponent')
    # parser.add_argument('--walklen', default=50, type=int, help='Walk length')
    parser.add_argument('--seed', default=0, type=int, help='Output directory')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.seed, args.npoints, args.alpha, args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
