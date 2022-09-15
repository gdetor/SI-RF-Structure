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
# This script computes the size and the center of mass for each classical
# receptive field described in [1].
import numpy as np
import matplotlib.pylab as plt


def area_of_activation(data):
    return sum(1 for i in data.flatten() if i > 0.0)


def rf_sub_size(Input, net_size, resolution):
    size = np.zeros((net_size*net_size,))
    coms = np.zeros((net_size*net_size, 2))
    R = np.zeros((net_size*resolution, net_size*resolution))
    Z = np.zeros((resolution, resolution))

    scale = 1.0/(resolution**2)

    X, Y = np.meshgrid(np.arange(Z.shape[0]), np.arange(Z.shape[1]))

    # count_roi, count_nroi, count_tot = 0, 0, 0
    plt.ion()
    for i in range(net_size):
        for j in range(net_size):
            Z = np.abs(Input[i, j, ...] * (Input[i, j, ...] > 0) +
                       0.0 * (Input[i, j, ...] < 0))

            R[i*resolution:(i+1)*resolution, j*resolution:(j+1)*resolution] = Z
            size[i*net_size+j] = area_of_activation(Z) * scale

            d = np.unravel_index(Z.argmax(), Z.shape)
            Z = np.roll(Z, Z.shape[0]/2-d[0], axis=0)
            Z = np.roll(Z, Z.shape[1]/2-d[1], axis=1)

            xc = ((Z*Y).sum()/Z.sum() - Z.shape[0]/2 + d[0])/float(Z.shape[0])
            yc = ((Z*X).sum()/Z.sum() - Z.shape[1]/2 + d[1])/float(Z.shape[1])

            coms[i*net_size+j, 0] = (xc+1.0) % 1
            coms[i*net_size+j, 1] = (yc+1.0) % 1
    return coms, R, size


def plot_rfs(size, C, Rx, Ry, color='b'):
    radius = np.sqrt(size[...]/np.pi)
    a, w = 0, C.shape[0]
    plt.scatter(Rx, Ry, s=15, color='w', edgecolor='k')
    plt.scatter(C[a:w, 1], C[a:w, 0], s=radius*500, alpha=0.4, color=color)
    plt.xticks([])
    plt.yticks([])


if __name__ == '__main__':
    net_size = 32
    resolution = 25

    # You have to modify the folders!!!
    folder = './data/Attention/IS/'
    Rx = np.load(folder+'gridxcoord.npy')
    Ry = np.load(folder+'gridycoord.npy')

    rfs = np.load(folder+'RFs.npy')

    C_e, R_e, size_e = rf_sub_size(rfs, net_size, resolution)
    print(sum(np.isnan(C_e[..., 0])))

    plt.figure(figsize=(8, 8))
    plot_rfs(size_e, C_e, Rx, Ry, 'b')
    plt.show()
