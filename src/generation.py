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
from numpy.random import multivariate_normal

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
def run_experiment(npoints, distrib, a, seed, outdir):
    info(inspect.stack()[0][3] + '()')

    random.seed(seed); np.random.seed(seed)
    if distrib == 'uniform':
        coords = np.random.rand(npoints, 2)
    elif distrib == 'normal':
        cov = np.eye(2) * .05
        coords = multivariate_normal([.5, .5], cov, npoints*2)[:npoints, :]

    plt.scatter(coords[:, 0], coords[:, 1])
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.savefig(pjoin(outdir, '{:03d}.png'.format(seed))); plt.close()

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

                newarc = False if (adj[walk[i], walk[i+1]] > 0) else True
                newvtx = False if (walk[i+1] in walk[:i+1]) else True
                symm = True if adj[walk[i+1], walk[i]] > 0 else False

                adj[walk[i], walk[i+1]] += 1

                ns[i+1] = ns[i] + 1 if newvtx else ns[i]
                ms[i+1] = ms[i] + 1 if newarc else ms[i]
                paired[i+1] = paired[i] + 2 if (newarc and symm) else paired[i]
                break
            bin0 = bin1
        assert j <= npoints

    ms[0] = 1 # Reciprocity when there is no arcs (avoid division by zero)
    rec = paired.astype(float) / ms
    ms[0] = 0 # Correct value
    ks = ms / ns

    import networkx as nx
    g = nx.from_numpy_matrix(adj)
    txt = '\n'.join(nx.generate_adjlist(g))
    open(pjoin(outdir, '{:03d}.adj'.format(seed)), 'w').write(txt)

    data = {}
    data['step'] = list(range(l+1))
    data['alpha'] = [a] * (l+1)
    data['walk'] = walk.tolist()
    data['vcount'] = ns.tolist()
    data['acount'] = ms.tolist()
    data['recipr'] = rec.tolist()
    data['k'] = ks.tolist()
    data['seed'] = [seed] * (l+1)

    return data

    # Plot
    # plot_walk1(walk, coords, outdir)
    # plot_walk2(walk, coords, outdir)

##########################################################
def main(npoints, distrib, a, nrealizations, seed, outdir):
    info(inspect.stack()[0][3] + '()')
    seeds = list(range(seed))
    data = {}
    for r in range(nrealizations):
        seed = r + seed
        ret = run_experiment(npoints, distrib, a, seed, outdir)
        if len(data) == 0:
            data = ret
        else:
            for k in data.keys():
                data[k] += ret[k]

    df = pd.DataFrame(data)
    csvpath = pjoin(outdir, 'results.csv')
    df.to_csv(csvpath, index=False)

##########################################################
if __name__ == "__main__":
    info(datetime.date.today())
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npoints', default=40, type=int, help='Number of points')
    parser.add_argument('--distrib', default='uniform', type=str, help='Distribution of points')
    parser.add_argument('--alpha', default=20., type=float, help='Factor of the exponent')
    parser.add_argument('--nrealizations', default=1, type=int, help='Number of realizations')
    parser.add_argument('--seed', default=0, type=int, help='Output directory')
    parser.add_argument('--outdir', default='/tmp/out/', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    readmepath = create_readme(sys.argv, args.outdir)

    main(args.npoints, args.distrib, args.alpha,
            args.nrealizations, args.seed, args.outdir)

    info('Elapsed time:{:.02f}s'.format(time.time()-t0))
    info('Output generated in {}'.format(args.outdir))
