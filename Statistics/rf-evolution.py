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
# This script computes the evolution of cRFs as it is described in [1].
import numpy as np
import matplotlib.pylab as plt

base = '/Users/gdetorak/Desktop/DNF-SOM/LTGM/Sizes/'


def evolution():
    mean, std = [], []
    for i in range(0, 6500, 50):
        size = np.load(base+'size_'+str(i)+'.npy')
        mean.append(np.mean(size))
        std.append(np.std(size))

    mean = np.array(mean, dtype='f')
    std = np.array(std, dtype='f')
    print mean.shape, std.shape

    np.save('rf_mean_ltgm', mean)
    np.save('rf_std_ltgm', std)

    return mean, std

if __name__ == '__main__':
    if 0:
        mu, sd = evolution()

    if 1:
        m1 = np.load('rf_mean.npy')
        m2 = np.load('rf_mean_ltgm.npy')
        mu = np.concatenate((m1, m2))

        sd1 = np.load('rf_std.npy')
        sd2 = np.load('rf_std_ltgm.npy')
        sd = np.concatenate((sd1, sd2))

        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)

        ax.plot(mu, 'b', ls='-', lw=3)
        ax.plot(mu+sd, 'r', ls='--', lw=2.5)
        ax.plot(mu-sd, 'r', ls='--', lw=2.5)
        ax.axis([0, 259, -.17, .20])
        ax.axvspan(xmin=125, xmax=140, ymin=0, ymax=1., color='black',
                   alpha=.15, zorder=10)
        ax.set_xticks([0, 130, 260])
        ax.set_xticklabels(('0', '6000', '12000'))
        ax.set_xlabel(r'Time (epochs)')
        ax.set_ylabel(r'$\mathbb{E}\{Size\}, (SD)$')
        plt.title(r'Evolution of RFs (mean and SD)')

        ax.annotate(r'Attention On', xy=(131, .025), xytext=(150, 0.10),
                    size = 14,
                    arrowprops=dict(arrowstyle='->',
                                    fc="0.6",
                                    connectionstyle="arc3,rad=.5")
                    )
        plt.savefig('evolution-rfs.pdf', frameon=False)
        plt.show()
