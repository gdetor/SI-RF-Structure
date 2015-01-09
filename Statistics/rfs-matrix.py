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
# This script computes the ncRFs using the regression method given by DiCarlo
# et al., 1998 and it's also fully explained in [1].
import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter

# data must be of dimension (n,n,m,m)
def highest_profile( data ):
	k, l = 32, 25
	prfl = np.zeros((k*k,l))
	for i in range( k ):
		for j in range( k ):
			index = np.unravel_index( np.argmax( data[i,j,...] ), (25,25) )[0]
			prfl[i*k+j,...] = data[i,j,index,...]
	return prfl

def rf_hist( data ):
	ii,jj = 0, 1
	x = np.zeros((2*data.shape[0],))
	y = np.zeros((2*data.shape[0],))
	for i in range( 0, 2*data.shape[0]-1, 2 ):
		y[i] = data[ii]
		y[i+1] = data[ii]
		if i%2==0:
			ii += 1

	for i in range( 1, 2*data.shape[0]-1 ):
		x[i] = jj
		x[i+1] = jj
		if i%2==0:
			jj += 1
	x[0], x[2*data.shape[0]-1] = 0, jj
	return x, y

def locate_noise( input ):
	n = input.shape[0]
	data = input.copy()

	count = 0
	for i in range( 1,n-1 ):
		for j in range( 1,n-1 ):
			if data[i,j] != 0:
				if np.sign(data[i+1,j])==np.sign(data[i,j]) and data[i+1,j] != 0:
					count += 1
				if np.sign(data[i-1,j])==np.sign(data[i,j]) and data[i-1,j] != 0:
					count += 1
				if np.sign(data[i,j-1])==np.sign(data[i,j]) and data[i,j-1] != 0:
					count += 1
				if np.sign(data[i,j+1])==np.sign(data[i,j]) and data[i,j+1] != 0:
					count += 1
				if count < 2:
					data[i,j] = 0
				count = 0
	return data

def snr( signal, sigma ):
	k = signal.shape[0]
	signal = signal.reshape(k,k)
	# Filtering the input signal
	filtered_s = gaussian_filter( signal, sigma )

	# Computing background noise
	noise = signal - filtered_s
	# Computing noise variance
	# noise_var = np.var( noise )
	noise_var = np.std( noise )

	# Computing signal and noise power
	signalPow = np.sum( signal**2 )/k
	noisePow = np.sum( noise**2 )/k

	# Computing snr and noise index
	snr = 10.0 * np.log10( signalPow/noisePow )
	noise_index = noise_var/np.abs(signal).max() * 100.0

	return snr, noise_index

def prettyFloat( float ):
	return '%.1f' % float

if __name__ == '__main__':
	n = 32
	folder = '/home/Local/SOM/Attention/REF/'
	RFs = np.load( folder+'RFs.npy' )

	prfl = np.zeros((8,25))

	ptr = np.random.randint(0,1014,(8,))

	index_u, index_l, sigma = 1, 9, 1.4
	upper, lower = 0, 0

	fig = plt.figure( figsize=(9.5,9) )
	for i in range( 8 ):
	 	plt.subplot(4,4,index_u,aspect=1)
		x, y = np.unravel_index( ptr[i], (n,n) )
		RF = RFs[x,y]
		d = np.unravel_index( RF.argmax(), (25,25) )
		RF = np.roll( RF, RF.shape[0]/2-d[0], axis=0 )
		RF = np.roll( RF, RF.shape[1]/2-d[1], axis=1 )
		RF = gaussian_filter( RF, sigma )
		abs_max = np.max( np.abs( RF ) )
		RF[np.where( ( ( RF < +0.10*abs_max ) & (RF>0) ) | ( ( RF > -0.10*abs_max ) & (RF < 0) ) ) ]=0
		RF = locate_noise( RF )
		d = np.unravel_index( RF.argmax(), (25,25) )
		prfl[i,...] = RF[d[0],...]
		lower, upper = min(lower,prfl.min()), max(upper,prfl.max())

		plt.imshow(RF, interpolation='bicubic', origin='lower',
			   	cmap=plt.cm.gray, extent=[0,10,0,10])
		plt.xticks([])
		plt.yticks([])
		# _, noise_index = snr( RF, sigma )
		# plt.title( str(prettyFloat(noise_index))+'%' )

		index_u += 1
		index_l += 1

	index_n = 9
	for i in range( 8 ):
	 	plt.subplot(4,4,index_n)
	 	X, Y = rf_hist( prfl[i] )
	 	plt.plot( X, Y, 'k', lw=1.5 )
	 	plt.axhline( 0, lw=1.5, c='k' )
	 	plt.axis([0,25,lower,upper])
	 	plt.xticks([])
	 	plt.yticks([])
	 	index_n += 1

	# plt.savefig( 'receptive-fields.pdf', dpi=100 )
	plt.show()
