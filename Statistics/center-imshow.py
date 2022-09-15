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
# This script computes the density diagrams of topographic maps. The method is
# given in [1].
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi


def imshow(Z, img, vmin, vmax):
    axis = plt.gca()

    plt.imshow(img,
               interpolation='bicubic',
               cmap=plt.cm.Purples,
               extent=[0, 1, 0, 1],
               origin='lower',
               vmin=vmin,
               vmax=vmax)
    axis.scatter(Z[:, 1], Z[:, 0], s=10, facecolors='w',
                 edgecolors='k', lw=.5, alpha=.85, zorder=2)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)


def diffuse(Z):
    Z = Z.reshape(n*n, 2)
    img = np.zeros((256, 256))
    for i in range(n*n):
        x, y = Z[i] * (256, 256)
        img[x, y] += 1
    return ndi.gaussian_filter(img, (12, 12), mode='wrap')


if __name__ == '__main__':
    n = 32
    folder = './data/Attention/'
    REF = np.load(folder+'REF/centers.npy').reshape(n*n, 2)
    LTGM = np.load(folder+'LTGM/centers.npy').reshape(n*n, 2)
    IS = np.load(folder+'IS/centers.npy').reshape(n*n, 2)
    LTGMIS = np.load(folder+'LTGM-IS/centers.npy').reshape(n*n, 2)

    REF_I = diffuse(REF)
    vmin, vmax = REF_I.min(), REF_I.max()
    LTGM_I = diffuse(LTGM)
    vmin, vmax = min(LTGM_I.min(), vmin), max(LTGM_I.max(), vmax)
    IS_I = diffuse(IS)
    vmin, vmax = min(IS_I.min(), vmin), max(IS_I.max(), vmax)
    LTGMIS_I = diffuse(LTGMIS)
    vmin, vmax = min(LTGMIS_I.min(), vmin), max(LTGMIS_I.max(), vmax)

    plt.figure(figsize=(15, 15))

    plt.subplot(2, 2, 1, aspect=1)
    imshow(REF, REF_I, vmin, vmax)

    plt.subplot(2, 2, 2, aspect=1)
    imshow(IS, IS_I, vmin, vmax)

    plt.subplot(2, 2, 3, aspect=1)
    imshow(LTGM, LTGM_I, vmin, vmax)

    plt.subplot(2, 2, 4, aspect=1)
    imshow(LTGMIS, LTGMIS_I, vmin, vmax)

    plt.show()
