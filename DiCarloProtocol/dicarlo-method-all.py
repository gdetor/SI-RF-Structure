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
# Computational reproduction of DiCarlo et al., 1998 experimental protocol.
# The model is explained in [1].
import numpy as np
from scipy import ndimage
from scipy.stats.stats import pearsonr
from scipy.ndimage import gaussian_filter


def thresholded(data, th):
    data = ndimage.gaussian_filter(data, sigma=2.)
    newData = data.copy()
    np.where(np.abs(newData) < th, newData, 0)
    return newData


def locate_noise(input):
    n = input.shape[0]
    data = input.copy()

    count = 0
    for i in range(1, n-1):
        for j in range(1, n-1):
            if data[i, j] != 0:
                if data[i+1, j] != 0 and np.sign(data[i+1, j]) == np.sign(data[i, j]):
                    count += 1
                if data[i-1, j] != 0 and np.sign(data[i-1, j]) == np.sign(data[i, j]):
                    count += 1
                if data[i, j-1] != 0 and np.sign(data[i, j-1]) == np.sign(data[i, j]):
                    count += 1
                if data[i, j+1] != 0 and np.sign(data[i, j+1]) == np.sign(data[i, j]):
                    count += 1
                if count < 2:
                    data[i, j] = 0
                count = 0
    return data


def threshold(RFs, size):
    p = 25
    Z = []
    for i in range(size):
        for j in range(size):
            RF = RFs[i, j, ...]
            RF = gaussian_filter(RF, sigma=.35)
            abs_max = np.max(np.abs(RF))
            RF[np.where(((RF < +0.10*abs_max) & (RF > 0)) |
                        ((RF > -0.10*abs_max) & (RF < 0)))] = 0
            RF = locate_noise(RF)
            exc = 100.0 * (RF > 0*abs_max).sum()/(p * p)
            inh = 100.0 * (RF < 0*abs_max).sum()/(p * p)
            Z.append([exc, inh])
    Z = np.array(Z)
    print(pearsonr(Z[:, 0], Z[:, 1]))
    return Z


if __name__ == '__main__':
    folder = './data/REF/'
    X = np.load(folder+'dots.npy')
    r = np.load(folder+'spike_trains.npy')
    m, n, k = 32, X.shape[0], X.shape[1]
    X = X.reshape(n, k*k)
    b = np.zeros((m*m, k*k))
    # b = np.zeros((m*m,k*k+1))

    # X = np.column_stack((np.ones((n,)),X))
    # b[:,0] = np.random.normal(0,.003,(m*m,))

    print(X.shape, r.shape)
    for i in range(m*m):
        b[i, ...] = np.dot(
            np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), r[i, ...])

    # np.save( folder+'real-rfs', b )
