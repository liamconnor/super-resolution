import sys, os

import matplotlib.pylab as plt
import numpy as np
import glob
import cv2
from scipy import signal
import optparse

try:
    from data_augmentation import elastic_transform
except:
    print("Could not load data_augmentation")

def Gaussian2D_v1(coords,  # x and y coordinates for each image.
                  amplitude=1,  # Highest intensity in image.
                  xo=0,  # x-coordinate of peak centre.
                  yo=0,  # y-coordinate of peak centre.
                  sigma_x=1,  # Standard deviation in x.
                  sigma_y=1,  # Standard deviation in y.
                  rho=0,  # Correlation coefficient.
                  offset=0):  # Offset from zero (background radiation).
    x, y = coords

    xo = float(xo)
    yo = float(yo)

    # Create covariance matrix
    mat_cov = [[sigma_x**2, rho * sigma_x * sigma_y],
               [rho * sigma_x * sigma_y, sigma_y**2]]
    mat_cov = np.asarray(mat_cov)
    # Find its inverse
    mat_cov_inv = np.linalg.inv(mat_cov)

    # PB We stack the coordinates along the last axis
    mat_coords = np.stack((x - xo, y - yo), axis=-1)

    G = amplitude * np.exp(-0.5*np.matmul(np.matmul(mat_coords[:, :, np.newaxis, :],
                                                    mat_cov_inv),
                                          mat_coords[..., np.newaxis])) + offset
    return G.squeeze()

def sim_sources(data, nsrc=250, noise=True):
    data = np.zeros_like(data, dtype=np.float32)
#    data = data.astype(np.uint16)
    nx, ny = data.shape
    for ii in range(nsrc):
        flux = np.random.randint(5,254)
        # Euclidean source counts
        flux = 50*np.random.uniform(0,1)**(-2/2.5)*5
        xind = np.random.randint(150//2, nx-150//2)
        yind = np.random.randint(150//2, ny-150//2)
        sigx = np.random.gamma(3,1.5)
        sigy = np.random.gamma(3,1.5)
#        flux /= (0.1*sigx*sigy)
        coords = np.meshgrid(np.arange(0, 150), np.arange(0, 150))
        source_ii = Gaussian2D_v1(coords,
                                amplitude=flux,
                                xo=150//2,
                                yo=150//2,
                                sigma_x=sigx,
                                sigma_y=sigy,
                                rho=np.random.uniform(-1,1),
                                offset=0)
        #source_ii += np.random.normal(1,0.5,source_ii.shape)
        data[xind-150//2:xind+150//2, yind-150//2:yind+150//2] += source_ii.T

    nbigblob = np.random.randint(0,5)

    for ii in range(nbigblob):
#        print("%d big blobs" % ii)
        # Euclidean source counts
        flux = 16*np.random.uniform(0,1)**(-2/2.5)*5
        xind = np.random.randint(150//2, nx-150//2)
        yind = np.random.randint(150//2, ny-150//2)
        sigx = np.random.normal(75,10)
        sigy = np.random.normal(75,10)
        coords = np.meshgrid(np.arange(0, nx), np.arange(0, ny))
        source_ii = Gaussian2D_v1(coords,
                                amplitude=flux,
                                xo=xind,
                                yo=yind,
                                sigma_x=sigx,
                                sigma_y=sigy,
                                rho=np.random.uniform(-1,1),
                                offset=0)
        #source_ii += np.random.normal(1,0.5,source_ii.shape)
        data += (source_ii.T)#.astype(np.uint16)

    if noise:
        noise_sig = 1e-1 * data.max()
        noise_arr = np.random.normal(0,noise_sig,data.shape)#.astype(np.uint16)
        noise_arr[noise_arr<0] = 0
        data += noise_arr

    return data#.astype(np.uint16)



def normalize_data(data, nbit=16):
    data = data - data.min()
    data = data/data.max()
    data *= (2**nbit-1)
    if nbit==16:
        data = data.astype(np.uint16)
    elif nbit==8:
        data = data.astype(np.uint8)
    return data

def convolvehr(data, kernel, plotit=False, 
               rebin=4, norm=True, nbit=8):
    if len(data.shape)==3:
        kernel = kernel[..., None]
        ncolor = 1
    else:
        ncolor = 3
       
    dataLR = signal.fftconvolve(data, kernel, mode='same')
    
    if norm is True:
         dataLR = normalize_data(dataLR, nbit=nbit)
         data = normalize_data(data, nbit=nbit)

    dataLR = dataLR[rebin//2::rebin, rebin//2::rebin]
    if plotit:
        plt.figure()
        dataLRflat = dataLR.flatten()
        dataLRflat = dataLRflat[dataLRflat!=0]
        dataflat = data.flatten()
        dataflat = dataflat[dataflat!=0]
        plt.hist(dataLRflat, color='C1', alpha=0.5, 
                 density=True, log=True, bins=255)
        plt.hist(dataflat, bins=255, color='C0', alpha=0.25, 
                 density=True, log=True)
        plt.title('Bit value distribution', fontsize=20)
        plt.xlabel('Pixel value')
        plt.ylabel('Number of pixels')
        plt.legend(['Convolved','True'])
        plt.figure()
        if norm is False:
            data = data.reshape(data.shape[0]//4,4,
                                data.shape[-2]//4, 4, 
                                ncolor).mean(1).mean(-2)
            plt.imshow(dataLR[..., 0], cmap='Greys', vmax=dataLR[..., 0].max()*0.1)
        else:
            plt.imshow(dataLR, vmax=dataLR[..., 0].max()*0.1)
        plt.title('Convolved', fontsize=15)
        plt.figure()
        if norm is False:
            plt.imshow(data[..., 0], cmap='Greys', vmax=data.max()*0.1)
        else:
            plt.imshow(data, vmax=data.max()*0.1)
        plt.title('True', fontsize=15)
        plt.figure()
        plt.imshow(kernel[...,0])
        plt.title('Kernel / PSF', fontsize=20)
        plt.show()
    return dataLR

def create_LR_image(fl, kernel, fdirout=None, 
                    pointsrcs=False, plotit=False, 
                    norm=True, sky=False, rebin=4, nbit=8, distort_psf=False):
    if type(fl) is str:
        fl = glob.glob(fl+'/*.png')
    elif type(fl) is list:
        pass
    else:
        print("Expected a list or a str as fl input")
        return

    if len(fl)==0:
        print("Input file list is empty")
        exit()

    for ii, fn in enumerate(fl):
        if ii%10==0:
            print("Finished %d/%d" % (ii, len(fl)))
        data = cv2.imread(fn)

        if pointsrcs:
            data = sim_sources(data[...,0], noise=False)[...,None]
            norm=True
        if sky:
            data = np.load('SKA-fun-model.npy')
            data = data[800:800+4*118, 800:800+4*124]
            mm=np.where(data==data.max())[0]
            data[data<0] = 0
            data /= (data.max()/255.0/12.)
            data[data>255] = 255
            data = data.astype(np.uint8)
            data = data[..., None]
        

        if distort_psf:
            kernel_ = kernel[..., None]*np.ones([1,1,3])
            kernel_ = elastic_transform(kernel_, alpha=np.random.randint(0,20), 
                                       sigma=3, alpha_affine=0)
            kernel_ = kernel_[..., 0]
            fdiroutPSF = fdirout[:-4]+'/psf/'
            print(fdiroutPSF)
            np.save(fdiroutPSF+fn.split('/')[-1][:-4] + '.npy', kernel_)
        else:
            kernel_ = kernel

        noise_arr = np.random.normal(0, 0.005*data.max(), data.shape)
        data += noise_arr
        dataLR = convolvehr(data, kernel_, plotit=plotit, 
                            rebin=rebin, norm=norm, nbit=nbit)

        data = normalize_data(data, nbit=nbit)
        dataLR = normalize_data(dataLR, nbit=nbit)

        if fdirout is None:
            fnout = fn.strip('.png')+'-conv.npy'
        else:
            fnout = fdirout + fn.split('/')[-1][:-4] + 'x%d.png' % rebin

        if nbit==8:
            cv2.imwrite(fnout, dataLR.astype(np.uint8))
        elif nbit==16:
            cv2.imwrite(fnout, dataLR.astype(np.uint16))

        if pointsrcs or sky:
            fnoutHR = fdirout + fn.split('/')[-1][:-4] + '.png'

            if nbit==8:
                cv2.imwrite(fnoutHR, data.astype(np.uint8))
            elif nbit==16:
                print('16')
                cv2.imwrite(fnoutHR, data.astype(np.uint16))

        del dataLR, data

if __name__=='__main__':
    # Example usage:
    # DIV2K: python hr2lr.py -d images/DIV2K_train_HR/ -k psf-briggs-2.npy -s 32 -o ./images/PSF-pointsrc-4x/test/ -p -r 4
    # Point sources: python hr2lr.py -d images/DIV2K_train_HR/ -k psf-briggs-2.npy -s 32 -o ./images/PSF-pointsrc-4x/test/ -p -r 4 -x
    # SKA sky image: python hr2lr.py -d images/DIV2K_train_HR/ -k psf-briggs-2.npy -s 64 -o ./images/PSF-pointsrc-4x/test/ -p --sky -r 2

    parser = optparse.OptionParser(prog="hr2lr.py",
                   version="",
                   usage="%prog input_dir kernel  [OPTIONS]",
                   description="Take high resolution images, convolve them, \
                   and save output.")

    parser.add_option('-d', dest='fdirin', type='str',
                      help="input directory")
    parser.add_option('-k', '--kernel', dest='kernel', type='str',
                      help="", default='Gaussian')
    parser.add_option("-s", "--ksize", dest='ksize', type=int,
                      help="size of kernel", default=64)
    parser.add_option('-o', '--fdout', dest='fdout', type='str',
                      help="output directory", default=None)
    parser.add_option('-p', '--plotit', dest='plotit', action="store_true",
                      help="plot")
    parser.add_option('-x', '--pointsrcs', dest='pointsrcs', action="store_true",
                      help="only do point sources")
    parser.add_option('--sky', dest='sky', action="store_true",
                      help="use SKA mid image as input")
    parser.add_option('-r', '--rebin', dest='rebin', type=int,
                      help="factor to spatially rebin", default=4)
    parser.add_option('-b', '--nbit', dest='nbit', type=int,
                      help="number of bits for image", default=8)
    parser.add_option('--scp', dest='scp', action="store_true",
                      help="scp data to cms-imaging")
    parser.add_option('--distort_psf', dest='distort_psf', action="store_true",
                      help="perturb PSF for each image generated")

    options, args = parser.parse_args()
    if options.kernel.endswith('npy'):
        kernel = np.load(options.kernel)
        nkern = len(kernel)
        kernel = kernel[nkern//2-options.ksize//2:nkern//2+options.ksize//2, 
                        nkern//2-options.ksize//2:nkern//2+options.ksize//2]
    elif options.kernel in ('Gaussian', 'gaussian'):
        kernel1D = signal.gaussian(8, std=1).reshape(8, 1)
        kernel = np.outer(kernel1D, kernel1D)

    fdirinTRAIN = options.fdirin+'/DIV2K_train_HR/'
    fdirinVALID = options.fdirin+'/DIV2K_valid_HR/'
    fdiroutTRAIN = options.fdout+'/train/'
    fdiroutVALID = options.fdout+'/valid/'
    fdiroutPSF = options.fdout+'/psf/'

    if not os.path.isdir(fdiroutTRAIN):
        print("Making output training directory")
        os.system('mkdir -p %s' % fdiroutTRAIN)

    if not os.path.isdir(fdiroutVALID):
        print("Making output validation directory")
        os.system('mkdir -p %s' % fdiroutVALID)

    if not os.path.isdir(fdiroutPSF):
        print("Making output PSF directory")
        os.system('mkdir -p %s' % fdiroutPSF)

    create_LR_image(fdirinTRAIN, kernel, fdirout=fdiroutTRAIN, 
            plotit=options.plotit, pointsrcs=options.pointsrcs, 
            sky=options.sky, rebin=options.rebin, nbit=options.nbit, 
            distort_psf=options.distort_psf)   
    create_LR_image(fdirinVALID, kernel, fdirout=fdiroutVALID, 
            plotit=options.plotit, pointsrcs=options.pointsrcs, 
            sky=options.sky, rebin=options.rebin, nbit=options.nbit,
            distort_psf=options.distort_psf)

    if not options.distort_psf:
        np.save('%s/psf.npy' % fdiroutPSF, kernel)

    if options.scp:
        fdirTRAINCMS = '/scratch/imaging/projects/dsa2000-sr/super-resolution/images-temp/train/'
        fdirVALIDCMS = '/scratch/imaging/projects/dsa2000-sr/super-resolution/images-temp/valid/'
        os.system('scp %s cms-imaging:%s' % (fdiroutTRAIN+'/*.png', fdirTRAINCMS))
        os.system('scp %s cms-imaging:%s' % (fdiroutVALID+'/*.png', fdirVALIDCMS))
        









