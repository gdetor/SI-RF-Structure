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
# This script reproduces the second figure of DiCarlo et al., 1998 using a
# computational method given in [1]. 
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import gaussian_filter
from numpy.fft import rfft2, ifftshift, irfft2


def extract(Z, position, shape, fill=0):
#    assert(len(position) == len(Z.shape))
#    if len(shape) < len(Z.shape):
#        shape = shape + Z.shape[len(Z.shape)-len(shape):]

    R = np.ones(shape, dtype=Z.dtype)*fill
    P  = np.array(list(position)).astype(int)
    Rs = np.array(list(R.shape)).astype(int)
    Zs = np.array(list(Z.shape)).astype(int)

    R_start = np.zeros((len(shape),)).astype(int)
    R_stop  = np.array(list(shape)).astype(int)
    Z_start = (P-Rs//2)
    Z_stop  = (P+Rs//2)+Rs%2

    R_start = (R_start - np.minimum(Z_start,0)).tolist()
    Z_start = (np.maximum(Z_start,0)).tolist()
    #R_stop = (R_stop - np.maximum(Z_stop-Zs,0)).tolist()
    R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
    Z_stop = (np.minimum(Z_stop,Zs)).tolist()

    r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
    z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]

    R[r] = Z[z]

    return R


# -----------------------------------------------------------------------------
def thresholded(data, threshold):
    return np.where(abs(data) < threshold, 0.0,data)

def locate_noise( input ):
    n = input.shape[0]
    data = input.copy()
    for i in range(1,n-1):
        for j in range(1,n-1):
            count = 0
            if data[i,j] != 0:
                if data[i+1,j] != 0 and np.sign(data[i+1,j])==np.sign(data[i,j]):
                    count += 1
                if data[i-1,j] != 0 and np.sign(data[i-1,j])==np.sign(data[i,j]):
                    count += 1
                if data[i,j-1] != 0 and np.sign(data[i,j-1])==np.sign(data[i,j]):
                    count += 1
                if data[i,j+1] != 0 and np.sign(data[i,j+1])==np.sign(data[i,j]):
                    count += 1
                if count < 2:
                    data[i,j] = 0
	return data

def cleanup(RF):
    size = RF.shape[0]
    #RF = gaussian_filter(RF, sigma=1.5)
    #threshold = 0.05*np.abs(RF.max())
    #RF = thresholded(RF.ravel(), threshold)
    #RF = locate_noise(RF.reshape(size,size))
    return RF


# -------------------------------------
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
    np.random.seed(137)

    # Standard units
    # --------------
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
    dots_number     = 750
    drum_length     = 250*mm
    drum_width      =  30*mm
    drum_shift      = 200*micrometer
    drum_velocity   = 40*mm / second
    simulation_time = 5*minute
    sampling_rate   = 5*ms
    dt              = sampling_rate
    skinpatch       = 10*mm,10*mm # width x height
    RF_sampling     = 25,25
    learning_som    = False
    learning        = False
    Rn = 16
    # R  = grid(Rn,noise=0.15)

    # Generate the drum pattern
    # -------------------------
    drum = np.zeros( (dots_number,2) )
    drum[:,0] = np.random.uniform(0,drum_length,dots_number)
    drum[:,1] = np.random.uniform(0,drum_width, dots_number)
    drum_x,drum_y = drum[:,0], drum[:,1]

    # Show the drum
    # -------------
    if 0:
        plt.figure(figsize = (16, 1+10 * drum_width/drum_length))
        plt.subplot(111,aspect=1)
        plt.scatter(drum_x, drum_y, s=10, facecolor='k', edgecolor='k')
        plt.xlim(0,drum_length)
        plt.xlabel("mm")
        plt.ylim(0,drum_width)
        plt.ylabel("mm")
        plt.show()

    print "Estimated number of samples: %d" % (simulation_time/dt)

    # SOM learning
    # -------------
    Sn = 32
    folder = '/home/Local/SOM/Attention/REF/'
    W = np.load( folder+'weights050000.npy' )
    R = np.zeros((Rn*Rn,2))
    R[:,0] = np.load( folder+'gridxcoord.npy' )
    R[:,1] = np.load( folder+'gridycoord.npy' )

    RF_count     = np.zeros((Sn,Sn,25,25))
    RF_sum       = np.zeros((Sn,Sn,25,25))
    global_count = np.zeros((Sn,Sn))
    global_sum   = np.zeros((Sn,Sn))

    scale = 960.0/(Sn*Sn)
    x_inf, x_sup, y_inf, y_sup = 0.0, 1.0, 0.0, 1.0
    X, Y = np.meshgrid( np.linspace(x_inf,x_sup,Sn+1,endpoint=True)[1:],
		        np.linspace(y_inf,y_sup,Sn+1,endpoint=True)[1:] )
    D = np.sqrt( (X-0.5)**2 + (Y-0.5)**2 )
    We = 3.65 * scale * h( D, 0.1 )
    Wi = 2.40 * scale * h( D, 1.0 )
    We_fft = rfft2( ifftshift( We[::-1,::-1] ) )
    Wi_fft = rfft2( ifftshift( Wi[::-1,::-1] ) )

    if learning:
        # Run the simulated drum
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
            I = generate_input(R,dots)

            # Generate the som answer
            V = dnf_response( Sn, Rn, I, W, We_fft, Wi_fft, 10.0, 25.0*.001 )

            # Compute the mean firing rate
            global_sum += V
            global_count += 1

            # Compute the local mean firing rate
            RF_sum   += V.reshape(Sn,Sn,1,1)*RF_mask
            RF_count += RF_mask

            # Display current skin patch dots and mask
            if 0:
                plt.figure(figsize=(10,10))
                plt.subplot(111,aspect=1)
                plt.scatter(dots[:,0],dots[:,1], s=50, facecolor='w', edgecolor='k')
                plt.xlim(0.0, 1.0)
                plt.ylim(0.0, 1.0)
                plt.show()


    mean = global_sum/(global_count+1)
    RFs = RF_sum/(RF_count+1) - mean.reshape(Sn,Sn,1,1)
    if learning: np.save( folder+"RFs.npy", RFs)
    RFs = np.load( folder+"RFs.npy")

    # Reconstitute the drum from model answers which does not make much sense
    # We should use the RF of a given neuron in fact and modulate according to
    # its answer or convolute the RF with current dot pattern
    if 1:
        Rc_y = (drum_length/skinpatch[0]) * Sn
        Rc_x = (drum_width/skinpatch[1]) * Sn
        Rc = np.zeros((Rc_x,Rc_y))

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
            I = generate_input(R,dots)
            # Generate the neural field answer
            V = dnf_response( Sn, Rn, I, W, We_fft, Wi_fft, 10.0, 25.0*.001 )

            x =  int((x/float(drum_length))*Rc_y)
            y =  int((y/float(drum_width))*Rc_x)
            Rc[y:y+Sn,x:x+Sn] = np.maximum(V,Rc[y:y+Sn,x:x+Sn])
            # Rc[y:y+Rn,x:x+Rn] += V

        # Compute y limit (we may have ended before end of drum)
        t = simulation_time
        z = t * drum_velocity
        x = z % (drum_length - skinpatch[0])
        ymax = int(z / (drum_length - skinpatch[0])) * drum_shift + skinpatch[0]


        plt.figure(figsize = (16, 1+10 * drum_width/drum_length))
        plt.subplot(111,aspect=1)

        plt.imshow(Rc, origin='lower', interpolation='bicubic', alpha=1,
                   cmap = plt.cm.gray_r, extent = [0, drum_length, 0, drum_width])
        plt.scatter(drum_x, drum_y, s=5, facecolor='w', edgecolor='k', alpha=.5)
        plt.xlim(0,drum_length)
        plt.xlabel("mm")
        #plt.ylim(0,drum_width)
        plt.ylim(0,ymax)
        plt.ylabel("mm")
        plt.show()




    # Show all RFs
    if 0:
        Z = np.zeros((Sn,25,Sn,25))
        for i in range(Sn):
            for j in range(Sn):
                RF = cleanup(RFs[i,j])
                # R = np.where(R<0, R/np.abs(R.min()),R/np.abs(R.max()))
                Z[i,:,j,:] = RF
        Z = Z.reshape(Sn*25,Sn*25)

        plt.figure(figsize=(14,10))
        plt.imshow(Z, interpolation='bicubic', origin='lower', cmap=plt.cm.PuOr_r, extent=(0,Sn,0,Sn))
        plt.colorbar()
        plt.xlim(0,Sn), plt.xticks(np.arange(Sn))
        plt.ylim(0,Sn), plt.yticks(np.arange(Sn))
        plt.grid()
        plt.title("Normalized Receptive fields", fontsize=16)
        plt.show()

    # Show a random RF
    if 0:
        i,j = np.random.randint(0,Sn,2)
        i,j = 8,8
        RF = cleanup(RFs[i,j])
        plt.figure(figsize=(8,6))
        plt.imshow(RF, interpolation='nearest', origin='lower',
                   cmap=plt.cm.gray_r, extent=[0,10,0,10])
        plt.colorbar()
        lmin = 0.50 * RF.min()
        lmax = 0.50 * RF.max()
        #CS = plt.contour(zoom(RF,10), levels=[lmin,lmax], colors='w',
        #                 origin='lower', extent=[0,10,0,10], linewidths=1, alpha=1.0)
        #plt.clabel(CS, inline=1, fontsize=12)
        plt.xlim(0,10), plt.xlabel("mm")
        plt.ylim(0,10), plt.ylabel("mm")
        plt.title("Normalized Receptive Field [%d,%d]" % (i,j), fontsize=16)
        plt.show()

    # Show excitatory/inhibitory ratio (scatter plot)
    if 0:
        matplotlib.rc('xtick', direction = 'out')
        matplotlib.rc('ytick', direction = 'out')
        matplotlib.rc('xtick.major', size = 8, width=1)
        matplotlib.rc('xtick.minor', size = 4, width=1)
        matplotlib.rc('ytick.major', size = 8, width=1)
        matplotlib.rc('ytick.minor', size = 4, width=1)

        Z = []
        for i in range(Sn):
            for j in range(Sn):
                p = 25
                RF = RFs[i,j]
                RF_max = np.abs(RF.max())
                #winner = np.unravel_index(np.argmax(RF), RF.shape)
                #RF = extract(RF,winner,(p,p))
                RF = cleanup(RFs[i,j])
                exc = 100 * ((RF >= +0.1*RF_max).sum()/ float(p*p))
                inh =  50 * ((RF <= -0.1*RF_max).sum()/ float(p*p))
                Z.append([exc,inh])
        Z = np.array(Z)
        X,Y = Z[:,0], Z[:,1]
        fig = plt.figure(figsize=(8,8), facecolor="white")
        ax = plt.subplot(1,1,1,aspect=1)
        plt.scatter(X+0.01,Y+0.01,s=5,color='k',alpha=0.25)


        # Show some points
        # I = [3,143,149,189,1,209,192,167,64,87,10,40,68,185,61,198]
        # plt.scatter(X[I],Y[I],s=5,color='k')
        # for i in range(len(I)):
        #     plt.annotate(" %c" % (chr(ord('A')+i)), (X[I[i]],Y[I[i]]), weight='bold')

        # Select some points by cliking them
        # letter = ord('A')
        # def onclick(event):
        #     global letter
        #     #print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
        #     #    event.button, event.x, event.y, event.xdata, event.ydata)
        #     C = (X-event.xdata)**2 + (Y-event.ydata)**2
        #     I = np.argmin(C)
        #     print I
        #     plt.ion()
        #     x,y = X[I],Y[I]
        #     plt.scatter(x,y,s=5,color='k')
        #     plt.annotate(" %c" % (chr(letter)), (x,y), weight='bold')
        #     plt.ioff()
        #     letter = letter+1

        # cid = fig.canvas.mpl_connect('button_press_event', onclick)

        plt.xlabel("Excitatory area (mm2)")
        plt.ylabel("Inhibitory area (mm2)")
        plt.xlim(1,100)
        plt.ylim(1,100)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks([1,10,100], ['1','10','100'])
        plt.yticks([1,10,100], ['1','10','100'])

        plt.plot([1,100],[1,100], ls='--', color='k')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.show()
