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

##########################################################
def main(outdir):
    """Short description"""
    info(inspect.stack()[0][3] + '()')

    n = 10
    coords = np.random.rand(n, 2)
    a = 0.5
    b = 1
    d = 0.5
    i0 = np.random.randint(n)
    tree = neighbors.KDTree(coords)
    inds, dists = tree.query_radius([coords[i0, :]], d,
                            return_distance=True, sort_results=True,)
    p = b / np.exp(a * d)

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
