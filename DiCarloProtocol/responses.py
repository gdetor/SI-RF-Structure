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
# -----------------------------------------------------------------------------
# Structure of Receptive Fields in Area 3b of Primary Somatosensory Cortex in
# the Alert Monkey - James J. DiCarlo, Kenneth O. Johnson, and Steven S. Hsiao
# The Journal of Neuroscience, April 1, 1998, 18(7):2626-2645
# -----------------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pylab as plt
from scipy.signal import fftconvolve
# from dana import convolve2d
from numpy.fft import rfft2, irfft2, ifftshift

# Some standard units
# -------------------
second          = 1.0
millisecond     = 1e-3 * second
ms              = millisecond
minute          = 60 * second
meter           = 1.0
millimeter      = 1e-3 * meter
mm              = millimeter
micrometer      = 1e-6 * meter

# --------------------------------------------------------- SkinPatch class ---
class SkinPatch:
        """ """

        def __init__(self, size=16*16, jitter=0.05):
                """ """

                n = np.sqrt(size)
                self.shape = (n,n)

                # Place receptors over the skin patch
                xmin, xmax = 0.0, 1.0
                ymin, ymax = 0.0, 1.0
                _X = (np.resize(np.linspace(xmin,xmax,n),(n,n))).ravel()
                _Y = (np.resize(np.linspace(ymin,ymax,n),(n,n)).T).ravel()
                X = _X + np.random.uniform(-jitter, jitter, n*n)
                Y = _Y + np.random.uniform(-jitter, jitter, n*n)
                Imin, Imax = np.argwhere(X < xmin), np.argwhere(X > xmax)
                while len(Imin) or len(Imax):
                        X[Imin] = _X[Imin] + np.random.uniform(-jitter, jitter, len(Imin))
                        X[Imax] = _X[Imax] + np.random.uniform(-jitter, jitter, len(Imax))
                        Imin, Imax = np.argwhere(X < xmin), np.argwhere(X > xmax)
                Imin, Imax = np.argwhere(Y < ymin), np.argwhere(Y > ymax)
                while len(Imin) or len(Imax):
                        Y[Imin] = _Y[Imin] + np.random.uniform(-jitter, jitter, len(Imin))
                        Y[Imax] = _Y[Imax] + np.random.uniform(-jitter, jitter, len(Imax))
                        Imin, Imax = np.argwhere(Y < ymin), np.argwhere(Y > ymax)
                self.X, self.Y = X.ravel(), Y.ravel()


        def __call__(self, S):
                """ Transform a set of stimulmi location into proper input """

                shape = self.shape
                S = np.array(S)
                X,Y = self.X, self.Y
                dX = np.abs(X.reshape(1,len(X)) - S[:,0].reshape(len(S),1))
                dY = np.abs(Y.reshape(1,len(Y)) - S[:,1].reshape(len(S),1))
                C = np.sqrt(dX*dX+dY*dY) / np.sqrt(2)
                sigma = 0.1
                return np.exp(-C**2/sigma**2).reshape(len(S),shape[0]*shape[1])

def dnf_response( n, Rn, stimulus, w, we, wi, time, dt ):
		alpha, tau = 0.1, 1.0
		U  = np.random.random((n,n)) * .01
		V  = np.random.random((n,n)) * .01

		V_shape = np.array(V.shape)

		# Computes field input accordingly
		D = (( np.abs( w - stimulus )).sum(axis=-1))/float(Rn*Rn)
		I = ( 1.0 - D.reshape(n,n) ) * alpha

		for j in range( int(time/dt) ):
				Z = rfft2( V * alpha )
				Le = irfft2( Z * we, V_shape).real
				Li = irfft2( Z * wi, V_shape).real
				U += ( -U + ( Le - Li ) + I )* dt * tau
				V = np.maximum( U, 0.0 )
		return V

def h(x, sigma=1.0):
        return np.exp(-0.5*(x/sigma)**2)

# -----------------------------------------------------------------------------
if __name__ == '__main__':

        # Seed for reproductibility
        # -------------------------
        np.random.seed(123)

        folder = '/home/Local/SOM/Attention/REF/'
        RF = np.load( folder+'receptive_field_2016.npy' ).reshape(16,16)
        events = np.load( folder+'one-neurons-spikes.npy')[0:120000]

        # Simulation parameters
        # ---------------------
        Rn              = 16  # Receptors numbers on skin patch (Rn x Rn)
        Sn              = 32  # SOM size (Sn x Sn)
        Rfn             = 25  # Receptive field sampling (Rfn x Rfn)
        drum_dots       = 750
        drum_length     = 250*mm
        drum_width      =  30*mm
        drum_shift      = 200*micrometer
        drum_velocity   = 40*mm / second
        sim_time        = 5*minute
        sim_dt          = 5*ms
        skinpatch_size  = 10*mm,10*mm

        patch = SkinPatch(Rn*Rn)
        folder = '/home/Local/SOM/Attention/REF/'
        W = np.load( folder+'weights050000.npy' )

        R = np.zeros((Rn*Rn,2))
        R[:,0] = np.load( folder+'gridxcoord.npy' )
        R[:,1] = np.load( folder+'gridycoord.npy' )

        scale = 960.0/(Sn*Sn)
        x_inf, x_sup, y_inf, y_sup = 0.0, 1.0, 0.0, 1.0
        X, Y = np.meshgrid( np.linspace(x_inf,x_sup,Sn+1,endpoint=True)[1:],
        	                	        np.linspace(y_inf,y_sup,Sn+1,endpoint=True)[1:] )
        D = np.sqrt( (X-0.5)**2 + (Y-0.5)**2 )
        We = 3.73 * scale * h( D, 0.1 )
        Wi = 2.40 * scale * h( D, 1.0 )
        We_fft = rfft2( ifftshift( We[::-1,::-1] ) )
        Wi_fft = rfft2( ifftshift( Wi[::-1,::-1] ) )

        # Generate the drum
        # -----------------
        if not os.path.exists("drum.npy"):
                drum = np.zeros( (drum_dots,2) )
                drum[:,0] = np.random.uniform(0, drum_length, drum_dots)
                drum[:,1] = np.random.uniform(0, drum_width, drum_dots)
                np.save("drum.npy", drum)
        else:
                drum = np.load("drum.npy")

        xdrum, ydrum = drum[:,0], drum[:,1]

        # Compute answer on the whole drum for a specific neuron
        # First version, using dots
        # ------------------------------------------------------
        X,Y,S = [], [], []
        # 10 * sim_time : we want to be sure to get the whole drum

        # events = []
        ii = 0
        Rc_y = (drum_length/skinpatch_size[0]) * Rn
        Rc_x = (drum_width/skinpatch_size[1]) * Rn
        Rc = np.zeros((Rc_x,Rc_y))
        for t in np.arange(0.0,10*sim_time, sim_dt):
		z = t * drum_velocity
                x = z % (drum_length - skinpatch_size[0])
		y = int(z / (drum_length - skinpatch_size[0])) * drum_shift

		# Maybe this should be adjusted since a stimulus lying outside the skin
		# patch may still have influence on the input (for example, if it lies
		# very near the border)
		xmin, xmax = x, x+skinpatch_size[0]
		ymin, ymax = y, y+skinpatch_size[1]

		if ymax > drum_width:
			break

		# Get dots contained on the skin patch (and normalize coordinates)
		dots = drum[(xdrum > xmin) * (xdrum < xmax) * (ydrum > ymin) * (ydrum < ymax)]
		dots -= (x,y)
		dots /= skinpatch_size[0], skinpatch_size[1]

		# Transform dot position into some input
		P = patch(dots).max(axis=0).reshape(1,Rn*Rn)

	    	if ii < 120000:
			if events[ii]==1:
				V = fftconvolve(RF,P.reshape(Rn,Rn),mode='same')
			        # V = convolve2d(P, RF, toric='True' ).reshape(Rn,Rn)
			        V = ( V > 0 )*V + ( V < 0 )*0

			     	x =  int((x/float(drum_length))*Rc_y)
				y =  int((y/float(drum_width))*Rc_x)

				# Rc[y:y+Rn,x:x+Rn] = np.maximum(V,Rc[y:y+Rn,x:x+Rn])
				Rc[y:y+Rn,x:x+Rn] += V
			ii += 1

		# Compute (simulated) activity in SOM
		# V = dnf_response( Sn, Rn, P, W, We_fft, Wi_fft, 10.0, 25.0*.001 )
	    	# if V[20,16] > 0:
	    	#        events.append(1)
	    	# else:
	    	#        events.append(0)

	# events = np.asarray(events)
	# np.save( folder+'one-neurons-spikes', events )
	np.save( folder+'convolution-rfs-dots', Rc )

	plt.figure(figsize = (16, 1+10 * drum_width/drum_length))
	plt.imshow(Rc, origin='lower', interpolation='bicubic', alpha=1,
			      cmap = plt.cm.Blues, extent = [0, drum_length, 0, drum_width])
	plt.xlim(0,drum_length)
	plt.xlabel("mm")
	plt.ylim(0,drum_width)
	plt.ylabel("mm")

	plt.show()
