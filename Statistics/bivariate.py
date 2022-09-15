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
# This script illustrated the bivariate plot presented in [1].

# import matplotlib.patheffects as PathEffects
# from scipy.ndimage import zoom
# import math

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('macosx')

matplotlib.rc('xtick', direction='out')
matplotlib.rc('ytick', direction='out')
matplotlib.rc('xtick.major', size=8, width=1)
matplotlib.rc('xtick.minor', size=4, width=1)
matplotlib.rc('ytick.major', size=8, width=1)
matplotlib.rc('ytick.minor', size=4, width=1)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', serif='Times')

# indices = [(3, 18), (26, 18), (10, 7), (25, 11), (3, 21), (8, 11),
#            (21, 14), (20, 16), (8, 19), (16, 5), (0, 9), (17, 15),
#            (7, 20), (20, 0), (27, 19), (4, 24)]

indices = [(10, 21), (29, 16), (28, 14), (20, 17), (13, 19), (3, 15), (23, 18),
           (0, 18), (8, 31), (16, 11), (0, 20), (24, 13), (11, 2), (1, 1),
           (19, 20), (2, 21)]


if __name__ == '__main__':
    Z = np.load('./data/areas-ref.npy')
    X, Y = Z[:, 0], Z[:, 1]

    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = plt.subplot(1, 1, 1, aspect=1)
    plt.scatter(X+0.01, Y+0.01, s=3, edgecolor='k', facecolor='k')

    # Show some points
    Index = [a*32+b for (a, b) in indices]
    # I = [3,143,149,189,1,209,192,167,64,87,10,40,68,185,61,198]
    plt.scatter(X[Index], Y[Index], s=5, color='k')
    for i in range(len(Index)):
        x, y = X[i], Y[i]
        letter = ord('A')+i
        plt.scatter(X[Index[i]],
                    Y[Index[i]],
                    s=40,
                    facecolor='None',
                    edgecolor='k')
        # label = plt.annotate(" %c" % (chr(letter)),
        #                      (x+.25, y+.25),
        #                      weight='bold',
        #                      fontsize=16,
        #                      path_effects=[PathEffects.withStroke(linewidth=2,\
        #                             foreground="w", alpha=.75)])
        plt.annotate(" %c" % (chr(ord('A')+i)),
                     (X[Index[i]]+.25, Y[Index[i]]+.25),
                     weight='bold')

    # Select some points by cliking them
    # letter = ord('A')
    # def onclick(event):
    #     global letter
    #     #print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
    #     #    event.button, event.x, event.y, event.xdata, event.ydata)
    #     C = (X-event.xdata)**2 + (Y-event.ydata)**2
    #     I = np.argmin(C)
    #     plt.ion()
    #     x,y = X[I],Y[I]
    #     # print x, y, I, np.unravel_index(I,(32,32))
    #     print np.unravel_index(I,(32,32)), ",",
    #     plt.scatter(x, y, s=40, facecolor='None', edgecolor='k')
    #     label = plt.annotate(" %c" % (chr(letter)), (x+.25,y+.25), weight='bold', fontsize=16,
    #               path_effects=[PathEffects.withStroke(linewidth=2, foreground="w", alpha=.75)])
    #     #label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65 ))

    #     plt.ioff()
    #     letter = letter+1

    # cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.xlabel(r'Excitatory area (mm2)')
    plt.ylabel(r'Inhibitory area (mm2')
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([5, 10, 30], ['5', '10', '30'])
    plt.yticks([5, 10, 30], ['5', '10', '30'])

    plt.xlim(5, 30)
    plt.ylim(5, 30)

    plt.text(5.5, 26, "n = 1024")

    plt.plot([1, 100], [1, 100], ls='--', color='k')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.savefig('./data/bivariate.pdf', dpi=72)
    plt.show()
