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
# This script applies all the filters and cleaning techniques to the ncRFs. You
# have to use this script before any further statistical analysis to the data.
import numpy as np
from scipy.stats.stats import pearsonr
from scipy.stats.mstats import gmean
from scipy.ndimage import gaussian_filter
# from matplotlib import rc
# import matplotlib.pylab as plt


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


# Computing the area of the receptive fields according to Dicarlo's
# protocol described in article "Structure of Receptive Fields in area 3b...
def clear_data(RFs, n):
    p = 25
    Z = []
    Noise = np.load('./data/noise.npy').reshape(n*n, p, p)
    cRFs = np.zeros((n*n, p, p))
    for i in range(n):
        for j in range(n):
            RF = RFs[i, j, ...]

            # WARNING : Centering the RF
            s0, s1 = np.unravel_index(np.argmax(RF), RF.shape)
            RF = np.roll(RF, 13-s0, axis=0)
            RF = np.roll(RF, 13-s1, axis=1)
            # WARNING : Centering the RF

            # RF += Noise[i*n+j]
            # RF = gaussian_filter( RF, sigma=2.2 )

            RF += 1.5*Noise[i*n+j]
            RF = gaussian_filter(RF, sigma=1.5)

            abs_max = np.max(np.abs(RF))
            RF[np.where(((RF < +0.10*abs_max) & (RF > 0)) |
                        ((RF > -0.10*abs_max) & (RF < 0)))] = 0
            RF = locate_noise(RF)
            cRFs[i*n+j, ...] = RF
            exc = 50.0 * (RF > 0).sum()/(p * p)
            inh = 50.0 * (RF < 0).sum()/(p * p)
            Z.append([exc, inh])

    Z = np.array(Z)
    np.nan_to_num(Z)
    print('------ Excitatory ------- Inhibitory -------')
    print('Minimum :', Z[:, 0].min(), Z[:, 1].min())
    print('Maximum :', Z[:, 0].max(), Z[:, 1].max())
    print('Mean :', np.mean(Z[:, 0]), np.mean(Z[:, 1]))
    print('Mean :', np.mean(np.log10(Z[:, 0])), np.mean(np.log10(Z[:, 1])))
    print('SD : ', np.std(np.log10(Z[:, 0])), np.std(np.log10(Z[:, 1])))
    print('GMean :', gmean(Z[:, 0]), gmean(Z[:, 1]))
    print("Pearson cor: ", pearsonr(Z[:, 0], np.abs(Z[:, 1])))
    return Z, cRFs

# Computing the SNR of the receptive fields.


def snr(signal, sigma):
    k = signal.shape[0]
    # Filtering the input signal
    filtered_s = gaussian_filter(signal, sigma)

    # Computing background noise
    noise = signal - filtered_s
    # Computing noise variance
    noise_var = np.var(noise)

    # Computing signal and noise power
    signalPow = np.sum(signal**2)/k
    noisePow = np.sum(noise**2)/k

    # Computing snr and noise index
    snr = 10.0 * np.log10(signalPow/noisePow)
    noise_index = noise_var/np.abs(signal).max() * 100.0

    return snr, noise_index, filtered_s


# Main :p
if __name__ == '__main__':
    np.random.seed(137)

    RFs = np.load('./data/real-rfs-ref.npy').reshape(32, 32, 25, 25)
    n, size, bins = RFs.shape[0], RFs.shape[2], 70
    Z, cRFs = clear_data(RFs, n)

    np.save('./data/areas-ref', Z)
    np.save('./data/cleared-rfs', cRFs)
