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
# This script computes illustrates some of the ncRFS annotated by the user.
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

# from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from matplotlib.patches import Rectangle

if __name__ == '__main__':
    n, m = 32, 25
    x, y = 22, 25
    RFs = np.load('./data/cleared-rfs.npy').reshape(n, n, m, m)

    fg = 0.0, 0.0, 0.0
    bg = 1.0, 1.0, 1.0
    matplotlib.rcParams['ytick.major.size'] = 0
    matplotlib.rcParams['ytick.minor.size'] = 9
    matplotlib.rcParams['xtick.major.width'] = .5
    matplotlib.rcParams['ytick.major.width'] = 0
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    matplotlib.rcParams['font.size'] = 12.0
    matplotlib.rc('axes', facecolor=bg)
    matplotlib.rc('axes', edgecolor=fg)
    matplotlib.rc('xtick', color=fg)
    matplotlib.rc('ytick', color=fg)
    matplotlib.rc('figure', facecolor=bg)
    matplotlib.rc('savefig', facecolor=bg)

    plt.figure(figsize=(10, 10))
    # plt.subplots_adjust( wspace=0.7, hspace=0.7 )
    # indices = [ ( 3,18), ( 3,11), (29,16),
    #             ( 6, 1), (22,21), ( 2, 7),
    #             (19,26), ( 7,20), (21, 2)]

    # indices = [(3, 18), (26, 18), (10, 7), (25, 11), (3, 21), (8, 11),
    #            (21, 14), (20, 16), (8, 19), (16, 5), (0, 9), (17, 15),
    #            (7, 20), (20, 0), (27, 19), (4, 24)]

    # indices = [(a,b) for (a,b) in np.random.randint(0,32,(32,2))]

    # indices = [(2, 12), (5, 9), (1, 17), (9, 18), (2, 14), (31, 11), (2, 30),
    #            (5, 16), (12, 2), (9, 9), (24, 22), (24, 13), (23, 29),
    #            (30, 6), (19, 20), (24, 19)]

    indices = [(10, 21), (29, 16), (28, 14), (20, 17), (13, 19), (3, 15),
               (23, 18), (0, 18), (8, 31), (16, 11), (0, 20), (24, 13),
               (11, 2), (1, 1), (19, 20), (2, 21)]

    vmin = vmax = 0
    for i in range(4):
        for j in range(4):
            index = i*4+j
            y, x = indices[index]
            RF = RFs[y, x]
            vmin = min(vmin, RF.min())
            vmax = max(vmax, RF.max())

    for i in range(4):
        for j in range(4):
            index = i*4+j
            y, x = indices[index]

            RF = RFs[y, x]

#            s0,s1 = np.unravel_index(np.argmax(RF),RF.shape)
#            RF = np.roll(RF,12-s0,axis=0)
#            RF = np.roll(RF,12-s1,axis=1)

            vmin, vmax = RF.min(), RF.max()

            plt.subplot2grid((4, 4), (i, j), rowspan=1, colspan=1)
            plt.imshow(RF,
                       interpolation='nearest',
                       cmap=plt.cm.gray_r,
                       origin='lower',
                       vmin=vmin,
                       vmax=vmax)

            plt.axis([0, RFs.shape[2]-1, 0, RFs.shape[2]-1])
            plt.xticks([]), plt.yticks([])
            plt.text(1, 1,
                     '%c' % (ord('A')+index),
                     weight='bold',
                     fontsize=20,
                     color='w',
                     path_effects=[PathEffects.withStroke(linewidth=1.5,
                                                          foreground="k",
                                                          alpha=.5)])

    print(vmin, vmax)
    plt.savefig('./data/matrix-rfs.pdf', dpi=100)
    plt.show()
