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
# This script computes the root mean square error of the feed-forward weights,
# as it is described in [1].
import numpy as np
import matplotlib.pylab as plt

if __name__ == '__main__':
    rmse_ref = np.load('rmse.npy')
    rmse_is = np.load('rmse-is.npy')
    rmse_ltgm = np.load('rmse-ltgm.npy')
    rmse_ltgmis = np.load('rmse-ltgmis.npy')

    fig = plt.figure(figsize=(18, 5))

    ax = plt.subplot(121)
    n1 = len(rmse_ref)
    X = np.arange(n1)*50
    ax.plot(X, rmse_ref, 'k')
    ax.set_ylim(-0.05, 0.3)
    ax.set_xlim(X[0]-10, X[-1]+50)
    ax.text(0.5, 0.95, 'Root mean square error (RMSE) during initial training',
            horizontalalignment='center',
            verticalalignment='top',
            transform=ax.transAxes)
    ax.set_xticks([0, 10000, 20000, 30000])
    ax.set_xlabel("Epochs")

    target = 200
    base = '/Users/gdetorak/Desktop/DNF-SOM/REF/'
    fname = ['weights000000.npy', 'weights000500.npy', 'weights001000.npy',
             'weights005000.npy', 'weights010000.npy']
    # RF = np.random.uniform(0,1,(32,32))
    x_insets = [.1, .15, .20, .25, .30]
    epochs = [0, 500, 1000, 5000, 10000]
    for i in range(len(x_insets)):
        RF = np.roll(np.load(base+fname[i])[target].reshape(16, 16), 8)

        x = x_insets[i]
        e = epochs[i]
        ax1 = fig.add_axes([x+0.025, 0.60, 0.15, 0.15], aspect=1)
        ax1.set_xticks([]), ax1.set_yticks([])
        ax1.set_xlim(0, 1), ax1.set_ylim(0, 1)
        ax1.imshow(RF, interpolation='bicubic',
                   extent=[0, 1, 0, 1],
                   cmap=plt.cm.gray)
        ax1.set_title("Epoch %d" % e, fontsize=10)
        ax.scatter([e, ], [rmse_ref[e/50], ], s=20, facecolor='w', zorder=10)


    n2 = len(rmse_is)
    X = (n1+np.arange(n2))*50

    ax = plt.subplot(322)
    ax.plot(X, rmse_ltgm, c='.75', zorder=-10, lw=.5)
    ax.plot(X, rmse_ltgmis, c='.75', zorder=-10, lw=.5)
    ax.plot(X, rmse_is, c='k')

    ax.set_xlim(X[0], X[-1]+50)
    ax.set_ylim(0, 0.01)
    ax.set_yticks([0, 0.01])
    ax.set_xticks([])
    ax.text(0.025, 0.95, 'RMSE / Intensive training',
            horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes)
    ax.set_xticks([40000, 50000, 60000, 70000], ['', '', '', ''])

    ax = plt.subplot(324)
    ax.plot(X, rmse_is, c='.75', zorder=-10, lw=.5)
    ax.plot(X, rmse_ltgmis, c='.75', zorder=-10, lw=.5)
    ax.plot(X, rmse_ltgm, c='k')
    ax.set_xlim(X[0], X[-1]+50)
    ax.set_ylim(0, 0.01)
    ax.set_yticks([0, 0.01])
    ax.set_xticks([])
    ax.text(0.025, 0.95, 'RMSE / Modulation within RoI',
            horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes)
    ax.set_xticks([40000, 50000, 60000, 70000], ['', '', '', ''])

    ax = plt.subplot(326)
    ax.text(0.5, 0.5, "text")
    ax.plot(X, rmse_is, c='.75', zorder=-10, lw=.5)
    ax.plot(X, rmse_ltgm, c='.75', zorder=-10, lw=.5)
    ax.plot(X, rmse_ltgmis, c='k')

    ax.set_xlim(X[0], X[-1]+50)
    ax.set_ylim(0, 0.01)
    ax.set_yticks([0, 0.01])
    ax.set_xticks([40000, 50000, 60000, 70000])
    ax.text(0.025, 0.95, 'RMSE / Intensive + Modulation',
            horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes)
    ax.set_xlabel("Epochs")

    # ax.plot(rmse_is, 'r', lw=1.9)
    # ax.plot(rmse_ltgm, 'm', lw=1.9)
    # ax.plot(rmse_ltgmis, 'k', lw=1.9)
    # # plt.axis([-5, 700, -0.01, 0.5])
    # ax.set_xlim([-5, 700])
    # ax.set_ylim([-0.01, 0.35])
    # ax.set_xticks([0, 60, 350, 700])
    # ax.set_xticklabels(('0', '3000', '17500', '35000'))
    # ax.axvline(60, ls='--', c='k', lw=2, alpha=.7)

    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.spines['bottom'].set_position(('data', 0))
    # ax.yaxis.set_ticks_position('left')
    # ax.spines['left'].set_position(('data', 0))

    # ax.set_xlabel('Time (epochs)')
    # ax.set_ylabel('RMSE')

    # ax2 = plt.axes([.65, .6, .2, .2], axisbg='w')
    # ax2.plot(rmse_ref[300:500], 'b', lw=1.0)
    # ax2.plot(rmse_is[300:500], 'r', lw=1.0)
    # ax2.plot(rmse_ltgm[300:500], 'm', lw=1.0)
    # ax2.plot(rmse_ltgmis[300:500], 'k', lw=1.0)
    # ax2.set_xlim([0, 200])
    # ax2.set_ylim([0, 0.006])
    # ax2.set_xticks([0, 200])
    # ax2.set_xticklabels(('1500', '2500'))
    # # plt.setp(ax2, xticks=[], yticks=[])
    plt.savefig('rmse.pdf')
    plt.show()
