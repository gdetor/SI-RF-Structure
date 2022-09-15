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
# This script computes the evolution of the raw feed-forward weights as it is
# described in [1].
import numpy as np
import matplotlib.pylab as plt


def mad(x):
    """ Returns the median absolute deviation.

        **Parameters**

        x : double
            Data array
    """
    return 1.4826 * np.median(np.abs(x - np.median(x)))


if __name__ == '__main__':
    base = './data/REF/'
    m, n, q = 1024, 16, 256
    if 0:
        epochs = 35000
        mean, std, median, mmad = [], [], [], []
        for i in range(0, epochs, 50):
            W = np.load(base+'weights'+str('%06d' % i)+'.npy')

            mean.append(np.mean(np.mean(W, axis=1)))
            std.append(np.std(np.std(W, axis=1)))
            median.append(np.median(np.median(W, axis=1)))
            mmad.append(np.apply_along_axis(mad, 0, np.apply_along_axis(mad, 1,
                                                                        W)))

        mean = np.array(mean)
        std = np.array(std)
        median = np.array(median)
        mmad = np.array(mmad)

        np.save('./data/mean', mean)
        np.save('./data/std', std)
        np.save('./data/median', median)
        np.save('./data/mad', mmad)

    if 1:
        mean = np.load('./data/mean.npy')
        std = np.load('./data/std.npy')
        # median = np.load('median.npy')
        # mmad = np.load('mad.npy')

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        M, target = 200, 100
        ax.plot(mean[:M], 'b', lw=2)
        ax.plot(mean[:M]+std[:M], 'r', ls='--', lw=1.3)
        ax.plot(mean[:M]-std[:M], 'r', ls='--', lw=1.3)
        # ax.plot(median[:M], 'k', lw=2)
        # ax.plot(mmad[:M], 'm', lw=2)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Mean (SD)')
        plt.xticks([0, 50, 100, 150, 200],
                   ('0', '2500', '5000', '7500', '10000'))
        plt.title('Evolution of'+r'$\mathbb{E}\{w_f\}$')

        bx = plt.axes([.19, .70, .15, .15], axisbg='y')
        W = np.load(base+'weights000000.npy')[target].reshape(n, n)
        bx.imshow(W, interpolation='bicubic', cmap=plt.cm.Purples)
        plt.title(r'epoch=0')
        plt.setp(bx, xticks=[], yticks=[])

        cx = plt.axes([.27, .45, .15, .15], axisbg='y')
        W = np.load(base+'weights002500.npy')[target].reshape(n, n)
        cx.imshow(W, interpolation='bicubic', cmap=plt.cm.Purples)
        plt.title(r'epoch=2500')
        plt.setp(cx, xticks=[], yticks=[])

        dx = plt.axes([.63, .3, .15, .15], axisbg='y')
        W = np.load(base+'weights007500.npy')[target].reshape(n, n)
        dx.imshow(W, interpolation='bicubic', cmap=plt.cm.Purples)
        plt.title(r'epoch=7500')
        plt.setp(dx, xticks=[], yticks=[])

#        plt.savefig('evolution-mean-weights.pdf')
        plt.show()
