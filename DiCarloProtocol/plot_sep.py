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
# It plots the SEP diagram. 
import numpy as np
import matplotlib.pylab as plt

if __name__=='__main__':
	folder = '/home/Local/SOM/Attention/REF/'
	SEP = np.load( folder+'one-neurons-spikes.npy' )[0:120000].reshape(100,1200)
	Rc = np.load( folder+'convolution-rfs-dots.npy' )
	drum_length, drum_width = 250*1e-3, 30*1e-3

	# R = SEP.copy()
	# X = np.load( folder+'dot-patterns.npy').reshape(120000,16*16)
	# b = np.dot( np.dot( np.linalg.inv( np.dot( X.T, X ) ), X.T ), R )
	# np.save( folder+'receptive_field_2016' )

	plt.figure( figsize=(12,6) )
	plt.subplots_adjust( wspace=0.0, hspace=.0 )
	plt.subplot(2,1,1)
	plt.imshow(Rc, origin='lower', interpolation='bicubic', alpha=1,
			      cmap = plt.cm.gray_r, extent = [0, drum_length, 0, drum_width])
	plt.xlim(0,drum_length)
	plt.xlabel("mm")
	plt.ylim(0,drum_width)
	plt.ylabel("mm")

	plt.subplot(2,1,2)
	# plt.imshow( SEP, aspect=2,interpolation='nearest', cmap=plt.cm.gray_r,
	# 		vmin=0, vmax=1, origin='lower')
	plt.imshow(SEP, origin='lower', interpolation='bicubic', alpha=1,
			      cmap = plt.cm.gray_r, extent = [0, drum_length, 0, drum_width])
	plt.xlim(0,drum_length)
	plt.xlabel("mm")
	plt.ylim(0,drum_width)
	plt.ylabel("mm")

	# plt.savefig('SEP.pdf', dpi=100 )
	plt.show()
