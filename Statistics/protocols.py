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
# -----------------------------------------------------------------------------
# Structure of Receptive Fields in Area 3b of Primary Somatosensory Cortex in
# the Alert Monkey - James J. DiCarlo, Kenneth O. Johnson, and Steven S. Hsiao
# The Journal of Neuroscience, April 1, 1998, 18(7):2626-2645
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib


def g(x, sigma=0.1):
    return np.exp(-x**2/sigma**2)


def fromdistance(fn, shape, center=None, dtype=float):
    def distance(*args):
        d = 0
        for i in range(len(shape)):
            d += ((args[i]-center[i])/float(max(1, shape[i]-1)))**2
        return np.sqrt(d)/np.sqrt(len(shape))
    if center is None:
        center = np.array(list(shape))//2
    return fn(np.fromfunction(distance, shape, dtype=dtype))


def Gaussian(shape, center, sigma=0.5):
    def g(x):
        return np.exp(-x**2/sigma**2)
    return fromdistance(g, shape, center)


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # Seed for reproductibility
    # -------------------------
    np.random.seed(12345)

    # Standard units
    # --------------
    second = 1.0
    millisecond = 1e-3 * second
    ms = millisecond
    minute = 60 * second
    meter = 1.0
    millimeter = 1e-3 * meter
    mm = millimeter
    micrometer = 1e-6 * meter

    # Simulation parameters
    # ---------------------
    dots_number = 750
    drum_length = 250*mm
    drum_width = 30*mm
    drum_shift = 200*micrometer
    drum_velocity = 40*mm / second
    simulation_time = 30*second
    sampling_rate = 5*ms
    dt = sampling_rate
    skinpatch = 10*mm, 10*mm  # width x height

    # Generate the drum pattern
    # -------------------------
    drum = np.zeros((dots_number, 2))
    drum[:, 0] = np.random.uniform(0, drum_length, dots_number)
    drum[:, 1] = np.random.uniform(0, drum_width, dots_number)
    drum_x, drum_y = drum[:, 0], drum[:, 1]

    dots = []
    n = 0
    for t in np.arange(0.0, simulation_time, dt):
        z = t * drum_velocity
        x = z % (drum_length - skinpatch[0])
        y = int(z / (drum_length - skinpatch[0])) * drum_shift
        # Maybe this should be adjusted since a stimulus lying outside the skin
        # patch may still have influence on the input (for example, if it lies
        # very near the border)
        xmin, xmax = x, x+skinpatch[0]
        ymin, ymax = y, y+skinpatch[1]

        # Get dots contained on the skin patch (and normalize coordinates)
        d = drum[(drum_x > (xmin)) *
                 (drum_x < (xmax)) *
                 (drum_y > (ymin)) *
                 (drum_y < (ymax))]
        d -= (x, y)
        d /= skinpatch[0], skinpatch[1]
        dots.extend((d*5).tolist())
    dots = np.array(dots)
    n = len(dots)
    X, Y = dots[:, 0], dots[:, 1]

    plt.figure(figsize=(20, 8))

    ax = plt.subplot(132, aspect=1)
    ax.scatter(X, Y, s=1, edgecolor='none', facecolor='k')
    ax.set_xlim(0, 5)
    ax.set_xlabel("mm")
    ax.set_ylim(0, 5)
    ax.set_ylabel("mm")
    ax.set_title("Drum Protocol (30 seconds), %d stimuli" % n)
    ax.text(0.25, 4.75, 'B', weight='bold', fontsize=32, color='k',
            ha='left', va='top')

    ax = plt.subplot(131, aspect=1)
    X = np.random.uniform(0, 5, 50000)
    Y = np.random.uniform(0, 5, 50000)
    ax.scatter(X, Y, s=1, edgecolor='none', facecolor='k')
    ax.set_xlim(0, 5)
    ax.set_xlabel("mm")
    ax.set_ylim(0, 5)
    ax.set_ylabel("mm")
    ax.set_title("Training Protocol, %d stimuli" % 50000)
    ax.text(0.25, 4.75, 'A', weight='bold', fontsize=32, color='k',
            ha='left', va='top')

    ax = plt.subplot(133, aspect=1)
    XY = np.zeros((25000/2, 2))
    d = 5.0/4
    for i in range(len(XY)):
        x, y = np.random.uniform(0, 5, 2)
        while d < x < 5-d and d < y < 5-d:
            x, y = np.random.uniform(0, 5, 2)
        XY[i] = x, y
    X, Y = XY[:, 0], XY[:, 1]
    ax.scatter(X, Y, s=1, edgecolor='none', facecolor='k')
    ax.text(0.25, 4.75, 'C', weight='bold', fontsize=32, color='k',
            ha='left', va='top')

    X = d+np.random.uniform(0, 2.5, 25000/2)
    Y = d+np.random.uniform(0, 2.5, 25000/2)
    ax.scatter(X, Y, s=1, edgecolor='none', facecolor='k')

    ax.set_xlim(0, 5)
    ax.set_xlabel("mm")
    ax.set_ylim(0, 5)
    ax.set_ylabel("mm")
    ax.set_title("RoI Protocol, %d stimuli" % 25000)

    plt.savefig("./data/protocols.png", dpi=200)
    plt.show()
