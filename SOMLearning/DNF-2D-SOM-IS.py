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
# DNF-2D-SOM-REF.py generates the topographic maps of during an intensive
# training of area 3b as it is described in [1].
import math as mt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy.fft import rfft2, irfft2, ifftshift

rc('text', usetex=True)
rc('font', family='serif')


def g(x, sigma=1.0):
    return np.exp(-0.5*(x/sigma)**2)


def prettyfloat(float):
    return '%.3f' % float


def print_parameters(n, Rn, Ke, sigma_e, Ki, sigma_i, T, dt, tau,
                     R_noise, epochs):
    print('Net size: ', n, 'x', n, 'x', Rn, 'x', Rn)
    print('Ke:', prettyfloat(Ke), 'sigma_e:', prettyfloat(sigma_e))
    print('Ki:', prettyfloat(Ki), 'sigma_i:', prettyfloat(sigma_i))
    print('Time:', prettyfloat(T), 'dt:', prettyfloat(dt),)
    print('tau:', prettyfloat(tau))
    print('Noise:', prettyfloat(R_noise), 'Epochs:', epochs)


def test_plot_fun(data):
    plt.cla(), plt.clf()
    plt.imshow(data, interpolation='nearest', cmap=plt.cm.jet)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.draw()


if __name__ == '__main__':
    np.random.seed(137)

    # Parameters
    # --------------------------------------------
    Rn = 16          # Receptors count (Rn x Rn)
    R_noise = 0.05   # Receptors placement noise
    n = 32           # Neural field size (n x n)

    T = 10.0    # 90.0 No of Euler's time discretization
    ms = 0.001
    dt = 100.0 * ms
    lrate = 0.4      # 0.005 Learning rate
    alpha = 0.1     # Time constant
    tau = 1.00    # Synapse temporal decay
    epochs = 35000   # Number of training epochs

    W_min, W_max = 0.00, 1.00     # Weights min/max values for initialization
    Ke = 960.0/(n*n) * 3.72  # Strength of lateral excitatory weights
    sigma_e = 0.1                 # Extent of lateral excitatory weights
    Ki = 960.0/(n*n) * 2.40  # Strength of lateral inhibitory weights
    sigma_i = 1.0                 # Extent of lateral excitatory weights

    frequency = 2    # Frequency of RoI stimulation
    samples_size = Rn * Rn

    # Neural field setup
    # --------------------------------------------
    U = np.random.uniform(0.00, 0.01, (n, n))
    V = np.random.uniform(0.00, 0.01, (n, n))

    folder_base = './data/REF/'
    folder = './data/IS/'
    W = np.load(folder_base+'weights025000.npy')

    # FFT implementation
    # --------------------------------------------
    mean = 0.5
    x_inf, x_sup, y_inf, y_sup = 0.0, 1.0, 0.0, 1.0
    X, Y = np.meshgrid(np.linspace(x_inf, x_sup, n+1)[1:],
                       np.linspace(y_inf, y_sup, n+1)[1:])
    Dist = np.sqrt((X-mean)**2 + (Y-mean)**2)

    We = Ke * g(Dist, sigma_e) * alpha
    Wi = Ki * g(Dist, sigma_i) * alpha

    print_parameters(n, Rn, Ke, sigma_e, Ki, sigma_i, T, dt, tau,
                     R_noise, epochs)

    We_fft = rfft2(ifftshift(We[::-1, ::-1]))
    Wi_fft = rfft2(ifftshift(Wi[::-1, ::-1]))

    print_parameters(n, Rn, Ke, sigma_e, Ki, sigma_i, T, dt, tau, R_noise,
                     epochs)

    # Skin receptors setup
    # --------------------------------------------
    size = epochs
    R = np.zeros((Rn*Rn, 2))
    R[:, 0] = np.load(folder_base+'gridxcoord.npy')
    R[:, 1] = np.load(folder_base+'gridycoord.npy')

    # Samples generation
    # --------------------------------------------
    S = np.random.uniform(0, 1, (size, 2))
    dX = np.abs(R[:, 0].reshape(1, Rn*Rn) - S[:, 0].reshape(size, 1))
    dX = np.minimum(dX, 1-dX)
    dY = np.abs(R[:, 1].reshape(1, Rn*Rn) - S[:, 1].reshape(size, 1))
    dY = np.minimum(dY, 1-dY)
    samples = np.sqrt(dX*dX+dY*dY)/mt.sqrt(2.0)
    samples = g(samples, 0.08)

    # Samples generation in RoI
    S = np.random.uniform(.25, .75, (size, 2))
    dX = np.abs(R[:, 0].reshape(1, Rn*Rn) - S[:, 0].reshape(size, 1))
    dX = np.minimum(dX, 1-dX)
    dY = np.abs(R[:, 1].reshape(1, Rn*Rn) - S[:, 1].reshape(size, 1))
    dY = np.minimum(dY, 1-dY)
    samples_roi = np.sqrt(dX*dX+dY*dY)/mt.sqrt(2.0)
    samples_roi = g(samples_roi, 0.08)

    # Actual training
    # --------------------------------------------
    # plt.ion()
    for e in range(epochs):
        # Pick a random sample
        if e % frequency == 0:
            stimulus = samples_roi[e]
        else:
            stimulus = samples[e]

        # Computes field input accordingly
        D = ((np.abs(W - stimulus)).sum(axis=-1))/float(Rn*Rn)
        Input = (1.0 - D.reshape(n, n)) * alpha

        # Field simulation until convergence
        for _ in range(int(T/dt)):
            Z = rfft2(V)
            Le = irfft2(Z * We_fft, (n, n)).real
            Li = irfft2(Z * Wi_fft, (n, n)).real
            U += (-U + (Le - Li) + Input) * tau * dt
            V = np.maximum(U, 0)

        # test_plot_fun( V )

        # Learning
        # --------
        W -= lrate * (Le.ravel() * (W - stimulus).T).T

        if e % 50 == 0:
            print("Epoch: %d" % e)
            # np.save(folder+'weights'+str('%06d' % e), W)

        # Field activity reset
        # --------------------
        U = np.random.uniform(0.00, 0.01, (n, n))
        V = np.random.uniform(0.00, 0.01, (n, n))

    np.save(folder+'weights'+str('%06d' % epochs), W)

    m = Rn
    plt.figure(figsize=(10, 10))    # 13, 7
    ax = plt.subplot(111, aspect=1)
    R = np.zeros((n*m, n*m))
    for j in range(n):
        for i in range(n):
            R[j*m:(j+1)*m, i*m:(i+1)*m] = W[j*n+i].reshape(m, m)
    im = plt.imshow(R, interpolation='nearest', cmap=plt.cm.bone_r,
                    vmin=0, vmax=1)
    plt.xticks(np.arange(0, n*m, m), [])
    plt.yticks(np.arange(0, n*m, m), [])
    plt.grid()
    plt.show()
