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
# This script generates the spike trains for each neuron of the neural field
# model given in [1], when it reproduces the protocol of Dicarlo et al. 1998.
# Use this script with the spike_events.py, responses.py and plot_set.py
# scripts.
import numpy as np
import matplotlib.pylab as plt

if __name__ == '__main__':
    folder = './data/REF/'
    response = np.load(folder+'activity_total.npy')
    print(response.shape)
    n, m = 32, response.shape[0]
    response = response.reshape(m, n, n)

    neurons_map = np.zeros((n*n, m))
    spikes_map = np.zeros((n*n, m))
    for i in range(n):
        for j in range(n):
            for k in range(m):
                neurons_map[i*n+j, k] = response[k, i, j]
                # if response[k,i,j] > 0:
                # 	spikes_map[i*n+j,k] = 1

    # np.savetxt( folder+'spike_trains.dat', neurons_map )
    np.save(folder+'spike_trains.npy', neurons_map)

    plt.plot(np.linspace(0, 1, m), neurons_map)
    # plt.plot( spikes_map[6] )
    plt.ylim([0, 2])
    plt.xlim([0, m])
    plt.show()
