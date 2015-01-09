#!/bin/env python
# -*- coding: utf-8 -*-
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
#
# Computational reproduction of DiCarlo et al., 1998 experimental protocol. 
# The model is explained in [1].
#
# -----------------------------------------------------------------------------
# Structure of Receptive Fields in Area 3b of Primary Somatosensory Cortex in
# the Alert Monkey - James J. DiCarlo, Kenneth O. Johnson, and Steven S. Hsiao
# The Journal of Neuroscience, April 1, 1998, 18(7):2626-2645
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pylab as plt
from numpy.fft import rfft2, irfft2, ifftshift

# -----------------------------------------------------------------------------
def grid(n, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, noise=0.0):
    _X = (np.resize(np.linspace(xmin,xmax,n),(n,n))).ravel()
    _Y = (np.resize(np.linspace(ymin,ymax,n),(n,n)).T).ravel()
    X = _X + np.random.uniform(-noise, noise, n*n)
    Y = _Y + np.random.uniform(-noise, noise, n*n)
    Imin, Imax = np.argwhere(X < xmin), np.argwhere(X > xmax)
    while len(Imin) or len(Imax):
        X[Imin] = _X[Imin] + np.random.uniform(-noise, noise, len(Imin))
        X[Imax] = _X[Imax] + np.random.uniform(-noise, noise, len(Imax))
        Imin, Imax = np.argwhere(X < xmin), np.argwhere(X > xmax)
    Imin, Imax = np.argwhere(Y < ymin), np.argwhere(Y > ymax)
    while len(Imin) or len(Imax):
        Y[Imin] = _Y[Imin] + np.random.uniform(-noise, noise, len(Imin))
        Y[Imax] = _Y[Imax] + np.random.uniform(-noise, noise, len(Imax))
        Imin, Imax = np.argwhere(Y < ymin), np.argwhere(Y > ymax)
    Z = np.zeros((n*n, 2))
    Z[:,0], Z[:,1] = X.ravel(), Y.ravel()
    return Z

# Receptors regular grid. Jitter can be added.
def grid_toric(n, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, noise=0.0):
    x = np.linspace(xmin,xmax,n,endpoint=False)
    y = np.linspace(ymin,ymax,n,endpoint=False)
    X,Y = np.meshgrid(x,y)
    X += np.random.uniform(-noise, noise, (n,n))
    X = np.mod(X+1,1)
    Y += np.random.uniform(-noise, noise, (n,n))
    Y = np.mod(Y+1,1)
    return X.ravel(), Y.ravel()

def g(x,sigma = 0.1):
    return np.exp(-x**2/sigma**2)

def fromdistance(fn, shape, center=None, dtype=float):
    def distance(*args):
        d = 0
        for i in range(len(shape)):
            d += ((args[i]-center[i])/float(max(1,shape[i]-1)))**2
        return np.sqrt(d)/np.sqrt(len(shape))
    if center == None:
        center = np.array(list(shape))//2
    return fn(np.fromfunction(distance,shape,dtype=dtype))

def Gaussian(shape,center,sigma=0.5):
    def g(x):
        return np.exp(-x**2/sigma**2)
    return fromdistance(g,shape,center)

def generate_input(R,S):
    """
    Given a grid of receptors and a list of stimuli positions, return the
    corresponding input
    """
    if len(S):
        dX = np.abs(R[:,0].reshape(1,len(R)) - S[:,0].reshape(len(S),1))
        dY = np.abs(R[:,1].reshape(1,len(R)) - S[:,1].reshape(len(S),1))
        C = np.sqrt(dX*dX+dY*dY) / np.sqrt(2)
        return g(C).max(axis=0)
    return np.zeros(R.shape[0])

def h(x, sigma=1.0):
    return np.exp(-0.5*(x/sigma)**2)

def stimulus_detection( S ):
	ins = S[6:19,6:19].sum()
	out = S.sum() - ins
	print out, ins
	if ins > out:
		return 1
	else:
		return 0

def activity_area( data ):
	return sum( 1 for i in data.flatten() if i > 0 )

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(137)

    # Standard units
    # --------------
    alpha, tau = 0.1, 1.0
    second          = 1.0
    millisecond     = 1e-3 * second
    ms              = millisecond
    minute          = 60 * second
    meter           = 1.0
    millimeter      = 1e-3 * meter
    mm              = millimeter
    micrometer      = 1e-6 * meter

    # Simulation parameters
    # ---------------------
    dots_number     = 500
    drum_length     = 250*mm
    drum_width      =  30*mm
    drum_shift      = 200*micrometer
    drum_velocity   = 40*mm / second
    simulation_time = 5*minute
    sampling_rate   = 10*ms
    dt              = sampling_rate
    skinpatch       = 10*mm,10*mm # width x height
    RF_sampling     = 25,25
    plot            = False
    Rn = 16
    R  = grid(Rn,noise=0.05)

    # Generate the drum pattern
    # -------------------------
    drum = np.zeros( (dots_number,2) )
    drum[:,0] = np.random.uniform(0,drum_length,dots_number)
    drum[:,1] = np.random.uniform(0,drum_width, dots_number)
    drum_x,drum_y = drum[:,0], drum[:,1]

    print "Estimated number of samples: %d" % (simulation_time/dt)

    # SOM learning
    # -------------
    Sn = 32
    W = np.random.uniform(0,1,(Sn*Sn,Rn*Rn))

    RF_count     = np.zeros((Sn,Sn,25,25))
    RF_sum       = np.zeros((Sn,Sn,25,25))
    global_count = np.zeros((Sn,Sn))
    global_sum   = np.zeros((Sn,Sn))

    U = np.random.random((Sn,Sn)) * .01
    V = np.random.random((Sn,Sn)) * .01

    x_inf, x_sup, y_inf, y_sup = 0.0, 1.0, 0.0, 1.0
    X, Y = np.meshgrid( np.linspace(x_inf,x_sup,Sn+1,endpoint=True)[1:],
		        np.linspace(y_inf,y_sup,Sn+1,endpoint=True)[1:] )
    D = np.sqrt( (X-0.5)**2 + (Y-0.5)**2 )
    We = 3.65*960.0/(32*32) * h( D, 0.1 )
    Wi = 2.40*960.0/(32*32) * h( D, 1.0 )
    We_fft = rfft2( ifftshift( We[::-1,::-1] ) )
    Wi_fft = rfft2( ifftshift( Wi[::-1,::-1] ) )

    time = 10.0
    folder = '/home/Local/SOM/Attention/Dicarlo/'

    # Run the simulated drum
    ii = 0
    for t in np.arange(0.0,simulation_time,dt):
	    z = t * drum_velocity
	    x = z % (drum_length - skinpatch[0])
	    y = int(z / (drum_length - skinpatch[0])) * drum_shift

	    # Maybe this should be adjusted since a stimulus lying outside the skin
	    # patch may still have influence on the input (for example, if it lies
	    # very near the border)
	    xmin, xmax = x, x+skinpatch[0]
	    ymin, ymax = y, y+skinpatch[1]

	    # Get dots contained on the skin patch (and normalize coordinates)
	    dots = drum[(drum_x > (xmin)) *
			(drum_x < (xmax)) *
			(drum_y > (ymin)) *
			(drum_y < (ymax))]
	    dots -= (x,y)
	    dots /= skinpatch[0],skinpatch[1]

	    # Compute RF mask
	    RF_mask = np.zeros(RF_sampling)
	    for dot in dots:
		index = (np.floor(dot*RF_sampling)).astype(int)
		RF_mask[index[1],index[0]] = 1

	    # Compute corresponding input (according to receptors)
	    S = generate_input(R,dots)

	    # Generate the som answer
	    D = (( np.abs( W - S )).sum(axis=-1))/float(Rn*Rn)
	    I = ( 1.0 - D.reshape(Sn,Sn) ) * alpha

	    for jj in range( int(time/dt) ):
		    Z = rfft2( V * alpha )
		    Le = irfft2( Z * We_fft, (Sn,Sn) ).real
		    Li = irfft2( Z * Wi_fft, (Sn,Sn) ).real
		    U += ( -U + ( Le - Li ) + I )* dt * tau
		    V = np.maximum( U, 0.0 )

	    W -= 0.05 * ( Le.ravel() * ( W - S ).T ).T
	    if ii%50==0:
		    print ii
		    np.save( folder+'weights'+str( '%06d' % ii ), W )
	    # Compute the mean firing rate
	    global_sum += V
	    global_count += 1

	    # Compute the local mean firing rate
	    RF_sum   += V.reshape(Sn,Sn,1,1)*RF_mask
	    RF_count += RF_mask

	    U = np.random.random((Sn,Sn)) * .01
	    V = np.random.random((Sn,Sn)) * .01

	    mean = global_sum/(1+global_count)
	    RFs = RF_sum/(1+RF_count) - mean.reshape(Sn,Sn,1,1)
	    ii += 1

    np.save( folder+'weights'+str( '%06d' % ii ), W )
    np.save( folder+'RFs.npy', RFs)
    np.save( folder+'RF_sum', RF_sum )
    np.save( folder+'RF_count', RF_count )
