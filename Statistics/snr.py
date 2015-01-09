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
# This script computes the snr and noise index for the regression method
# described in [1].
import numpy as np
from scipy.ndimage import gaussian_filter


def snr(signal, sigma):
    k = signal.shape[0]
    # Filtering the input signal
    filtered_s = gaussian_filter(signal, sigma)

    # Computing background noise
    noise = signal - filtered_s
    # Computing noise variance
    noise_std = np.std(noise)

    # Computing signal and noise power
    signalPow = np.sum(signal**2)/k
    noisePow = np.sum(noise**2)/k

    # Computing snr and noise index
    snr = 10.0 * np.log10(signalPow/noisePow)
    # noise_index = noise_std/np.abs(filtered_s).max() *100.0
    noise_index = np.abs(filtered_s).max()/noise_std     # *100.0

    return snr, noise_index, filtered_s

if __name__ == '__main__':
    n, k = 32, 25
    # Change the folder!!!
    folder = '/home/Local/SOM/Attention/REF/'
    # RFs_mask = np.load( folder+'RFs.npy' ).reshape(n*n,k*k)
    RFs_regr = np.load(folder+'real-rfs.npy')

    snr_power = np.zeros((n*n, 2))
    rf_noise_index = np.zeros((n*n, 2))
    for i in range(n*n):
        # snr_power[i,0], rf_noise_index[i,0],_ = snr( RFs_mask[i], 1.5 )
        snr_power[i, 1], rf_noise_index[i, 1], _ = snr(RFs_regr[i], 2.5)
