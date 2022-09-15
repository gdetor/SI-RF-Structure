# Copyright (c) 2014, Georgios Is. Detorakis (gdetor@gmail.com) and
#                     Nicolas P. Rougier (nicolas.rougier@inria.fr)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# This file is part of the source code accompany the peer-reviewed article:
# [1] "Structure of Receptive Fields in a Computational Model of Area 3b of
# Primary Sensory Cortex", Georgios Is. Detorakis and Nicolas P. Rougier,
# Frontiers in Computational Neuroscience, 2014.
#
# This script computes the the histograms of ncRFs according to DiCarlo et
# al. 1998 (the method is described in [1]).
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pylab as plt

# from scipy.stats.stats import pearsonr
# from scipy.stats.mstats import gmean
# from scipy.ndimage import gaussian_filter

rc('text', usetex=True)
rc('font', serif='Times')

# Plotting histograms and bivariate diagrams.


def plot_area(Z, bins):
    X, Y = Z[:, 0], Z[:, 1]
    plt.figure(figsize=(20, 10))
    # plt.subplots_adjust( hspace=.7)

    ax = plt.subplot(3, 4, 5)
    plt.hist(X, bins=np.logspace(.1, 2, bins), color='k', alpha=.3)
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()
    plt.xlim(1, 100)
    plt.ylim(0, 250)
    plt.gca().set_xscale('log')
    plt.xticks([1, 10, 100], ('1', '10', '100'))
    plt.yticks([0, 100, 200, 250], ('0', '100', '200', ''))
    plt.xlabel(r'Area (mm$^2$)')
    plt.ylabel(r'Number of neurons')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.text(1, 255, 'Excitatory Area',
             ha='left',
             va='baseline')

    bx = plt.subplot(3, 4, 6)
    plt.hist(Y, bins=np.logspace(.1, 2, bins), color='k', alpha=.3)
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()
    plt.xlim(1, 100)
    plt.ylim(0, 250)
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.xticks([1, 10, 100], ('1', '10', '100'))
    plt.yticks([0, 100, 200, 250], ('', '', '', ''))
    plt.xlabel(r'Area (mm$^2$)')
    plt.gca().set_xscale('log')
    bx.spines['right'].set_color('none')
    bx.spines['top'].set_color('none')
    bx.xaxis.set_ticks_position('bottom')
    bx.yaxis.set_ticks_position('left')
    plt.xticks([1, 10, 100], ('1', '10', '100'))

    plt.text(1, 255, 'Inhibitory Area',
             ha='left',
             va='baseline')

    cx = plt.subplot(3, 4, 7)
    plt.hist((X+Y), bins=np.logspace(.1, 2, bins), color='k', alpha=.3)
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()
    plt.xlim(1, 100)
    plt.xticks([1, 10, 100], ('1', '10', '100'))
    plt.ylim(0, 250)
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.gca().set_xscale('log')
    plt.xticks([1, 10, 100], ('1', '10', '100'))
    plt.yticks([0, 100, 200, 250], ('', '', '', ''))

    plt.xlabel(r'Area (mm$^2$)')
    cx.spines['right'].set_color('none')
    cx.spines['top'].set_color('none')
    cx.xaxis.set_ticks_position('bottom')
    cx.yaxis.set_ticks_position('left')
    plt.xticks([1, 10, 100], ('1', '10', '100'))

    plt.text(1, 255, 'Total area',
             ha='left',
             va='baseline')

    dx = plt.subplot(3, 4, 8)
    plt.hist(X/Y, bins=np.logspace(-1, 1, bins), color='k', alpha=.3)
    plt.gca().get_xaxis().tick_bottom()
    plt.gca().get_yaxis().tick_left()
    plt.gca().set_xscale('log')
    plt.ylim(0, 250)
    plt.xlim(.1, 10)
    plt.yticks([0, 100, 200, 250], ('', '', '', ''))
    plt.xticks([.1, 1, 10], ('0.1', '1', '10'))
    plt.xlabel(r'Excitatory/Inhibitory area')
    dx.spines['right'].set_color('none')
    dx.spines['top'].set_color('none')
    dx.xaxis.set_ticks_position('bottom')
    dx.yaxis.set_ticks_position('left')
    plt.text(0.1, 255, 'Ratio',
             ha='left',
             va='baseline')


# Main :p
if __name__ == '__main__':
    np.random.seed(137)
    Z = np.load('./data/areas-ref.npy')
    bins = 70

    fg = (0.0, 0.0, 0.0)
    bg = (1.0, 1.0, 1.0)
    matplotlib.rc('xtick', direction='out')
    matplotlib.rc('ytick', direction='out')
    matplotlib.rc('xtick.major', size=8, width=1)
    matplotlib.rc('xtick.minor', size=4, width=1)
    matplotlib.rc('ytick.major', size=8, width=1)
    matplotlib.rc('ytick.minor', size=4, width=1)
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', serif='Times')
    matplotlib.rc('figure', facecolor=bg)
    matplotlib.rc('savefig', facecolor=bg)

    plot_area(Z, 70)
    plt.savefig('histograms.pdf', dpi=72)
    plt.show()
