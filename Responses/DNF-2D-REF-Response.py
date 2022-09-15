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
# DNF-2D-SOM-REF.py computes the responses of the model described in [1]
# in normal case.
'''
Model response. Calculates the response of the DNF-SOM-MultiDimInput
model.
'''
import math as mt
import numpy as np
from sys import stdout
import matplotlib.pyplot as plt
from numpy.fft import rfft2, irfft2, ifftshift


def progress_bar(width, percent):
    marks = mt.floor(width * (percent/100.0))
    spaces = mt.floor(width - marks)
    loader = '[' + ('#' * int(marks)) + (' ' * int(spaces)) + ']'

    stdout.write("%s %d%%\r" % (loader, percent))
    if percent >= 100:
        stdout.write("\n")
    stdout.flush()


def g(x, sigma=1.0):
    return np.exp(-0.5*(x/sigma)**2)


def area_of_activity(data):
    return sum(1 for i in data.flatten() if i > 0.0)


def dnfsom_activity(n, Rn, rf_size, tau, T, alpha, folder):
    ms = 0.001  # ms definition
    dt = 35.0 * ms   # Euler's time step

    # Files to be loaded
    filename = 'weights050000.npy'
    filenames = 'model_response_64_final'

    # Allocation of arrays and loading necessary files
    response = np.zeros((rf_size*n, rf_size*n))
    W = np.load(folder+filename)
    Rx = np.load(folder+'gridxcoord.npy')
    Ry = np.load(folder+'gridycoord.npy')
    V = np.random.random((n, n)) * .01
    U = np.random.random((n, n)) * .01

    # FFT implementation
    mean = 0.5
    Ke, Ki = 3.65, 2.40
    sigma_e, sigma_i = 0.1, 1.0
    x_inf, x_sup, y_inf, y_sup = 0.0, 1.0, 0.0, 1.0
    X, Y = np.meshgrid(np.linspace(x_inf, x_sup, n+1)[1:],
                       np.linspace(y_inf, y_sup, n+1)[1:])
    D = np.sqrt((X-mean)**2 + (Y-mean)**2)

    We = Ke * g(D, sigma_e) * alpha * 960.0/(n*n)
    Wi = Ki * g(D, sigma_i) * alpha * 960.0/(n*n)

    We_fft = rfft2(ifftshift(We[::-1, ::-1]))
    Wi_fft = rfft2(ifftshift(Wi[::-1, ::-1]))

    # Stimuli generation
    S = np.zeros((rf_size*rf_size, 2))
    for i, x in enumerate(np.linspace(0.0, 1.0, rf_size)):
        for j, y in enumerate(np.linspace(0.0, 1.0, rf_size)):
            S[i*rf_size+j, 0] = x
            S[i*rf_size+j, 1] = y
    dX = np.abs(Rx.reshape(1, Rn*Rn) - S[:, 0].reshape(rf_size*rf_size, 1))
    dX = np.minimum(dX, 1-dX)
    dY = np.abs(Ry.reshape(1, Rn*Rn) - S[:, 1].reshape(rf_size*rf_size, 1))
    dY = np.minimum(dY, 1-dY)
    samples = np.sqrt(dX*dX+dY*dY)/mt.sqrt(2.0)
    samples = g(samples, 0.08)

    # Calculation of model response
    step = 0
    jj = 100.0/(float(rf_size))
    for i in range(rf_size):
        for j in range(rf_size):
            D = ((np.abs(W - samples[i*rf_size+j])).sum(axis=-1))/float(Rn*Rn)
            Input = (1.0 - D.reshape(n, n)) * alpha

            for k in range(int(T/dt)):
                Z = rfft2(V)
                Le = irfft2(Z * We_fft, (n, n)).real
                Li = irfft2(Z * Wi_fft, (n, n)).real
                U += (-U + (Le - Li) + Input) * tau * dt
                V = np.maximum(U, 0)
            response[i*n:(i+1)*n, j*n:(j+1)*n] = V

            V = np.random.random((n, n)) * .01
            U = np.random.random((n, n)) * .01
        step += jj
        progress_bar(30, step)

    np.save(folder+filenames, response)
    plt.imshow(response, interpolation='bicubic', cmap=plt.cm.hot,
               extent=[0, rf_size*n, 0, rf_size*n])
    plt.xticks(np.arange(0, rf_size*n, n), [])
    plt.yticks(np.arange(0, rf_size*n, n), [])
    plt.show()


if __name__ == '__main__':
    np.random.seed(137)
    model_size = 32
    rf_resolution = 64
    num_receptors = 16
    T = 10.0
    tau = 1.0
    alpha = 0.1
    # Change the folder path!!!
    folder = './data/REF/'

    dnfsom_activity(model_size, num_receptors, rf_resolution, tau, T, alpha,
                    folder)
