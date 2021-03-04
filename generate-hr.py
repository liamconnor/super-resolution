import sys

import matplotlib
import numpy as np
import matplotlib.pylab as plt
import optparse
from scipy import signal

from model import resolve_single
from model.edsr import edsr
from utils import load_image, plot_sample
from model.wdsr import wdsr_b
from model.common import resolve_single16, tf
import hr2lr

# SKA
# python generate-hr.py images/PSF-nkern512-4x/train/0281x4.png ./weights-3e5-wdsr-b-32-x4.h5 -f images/PSF-nkern512-4x/train/0281.png --vm 150
# python generate-hr.py images/PSF-nkern512-4x/train/0281x4.png ./weights-psf256-noise-4x.h5 -f images/PSF-nkern512-4x/train/0281.png
# python generate-hr.py ./random-60x15s-nonoise.npy ./weights-psf256-noise-4x.h5 -k psf-briggs-2.npy -f ./random-60x15s-nonoise.npy --vm 25

def plotter(datalr, datasr, datahr=None, 
            cmap='viridis', suptitle=None, 
            fnfigout='test.pdf', vm=75, nbit=8, calcpsnr=True):

    fig=plt.figure(figsize=(10,7.8))

    datasr = datasr.numpy()
    # datalr = hr2lr.normalize_data(datalr, nbit=nbit)
    # datasr = hr2lr.normalize_data(datasr, nbit=nbit)

    if calcpsnr:
        psnr = tf.image.psnr(datasr[None, ..., 0, None].astype(np.uint16), 
                             datahr[None, ..., None].astype(np.uint16), 
                             max_val=2**(nbit)-1)
        ssim = tf.image.ssim(datasr[None, ..., 0, None].astype(np.uint16), 
                             datahr[None, ..., None].astype(np.uint16), 
                             2**(nbit)-1, filter_size=2, 
                             filter_sigma=1.5, k1=0.01, k2=0.03)
        psnr = "PSNR = %0.1f\nSSIM = %0.4f" % (psnr, ssim)


    if datahr is None:
        nsub=2
    else:
        nsub=3
    if datahr is not None:
      pass
        #datahr = hr2lr.normalize_data(datahr, nbit=nbit)

    # datalr = hr2lr.normalize_data(datalr, nbit=nbit)
    # datasr = hr2lr.normalize_data(datasr, nbit=nbit)
    ax1 = plt.subplot(2,nsub,1)
    plt.title('Convolved LR', color='C1', fontweight='bold', fontsize=15)
    plt.axis('off')
    plt.imshow(datalr[...,0], cmap=cmap, vmax=vm, vmin=0, 
               aspect='auto', extent=[0,1,0,1])
    plt.setp(ax1.spines.values(), color='C1')

    ax2 = plt.subplot(2,nsub,2, sharex=ax1, sharey=ax1)
    plt.title('Reconstructed HR', color='C2', 
              fontweight='bold', fontsize=15)
    plt.imshow(datasr[...,0], cmap=cmap, vmax=vm, vmin=0, 
              aspect='auto', extent=[0,1,0,1])
    plt.axis('off')
    if calcpsnr:
      plt.text(0.6, 0.85, psnr, color='white', fontsize=7, fontweight='bold')


    if nsub==3:
        ax5 = plt.subplot(2,nsub,3,sharex=ax1, sharey=ax1)
        plt.title('True HR', color='k', fontweight='bold', fontsize=15)
        plt.imshow(datahr, cmap=cmap, vmax=vm, vmin=0, aspect='auto', extent=[0,1,0,1])
        plt.axis('off')

    ax3 = plt.subplot(2,nsub,4)
    plt.axis('off')
    plt.xlim(0.25,0.45)
    plt.ylim(0.25,0.45)
    plt.imshow(datalr[:,:,0], cmap=cmap, vmax=vm, vmin=0, 
              aspect='auto', extent=[0,1,0,1])
    plt.title('Convolved LR \nzoom', color='C1', fontweight='bold', fontsize=15)

    ax4 = plt.subplot(2,nsub,5, sharex=ax3, sharey=ax3)
    plt.title('Reconstructed HR \nzoom ', color='C2', 
              fontweight='bold', fontsize=15)

    plt.imshow(datasr[:,:,0], cmap=cmap, 
              vmax=vm, vmin=0, aspect='auto', extent=[0,1,0,1])
    plt.axis('off')
    plt.xlim(0.25,0.45)
    plt.ylim(0.25,0.45)
    plt.suptitle(suptitle, color='C0', fontsize=20)

    if nsub==3:
        ax6 = plt.subplot(2,nsub,6,sharex=ax3, sharey=ax3)
        plt.title('True HR zoom', color='k', fontweight='bold', fontsize=15)        
        plt.imshow(datahr[:,:], cmap=cmap, 
                   vmax=vm, vmin=0, aspect='auto', extent=[0,1,0,1])
        plt.xlim(0.25,0.45)
        plt.ylim(0.25,0.45)
        plt.axis('off')
    else:
        plt.axis('off')

    plt.savefig(fnfigout)        
    plt.show()

def func(fn_img, fn_model, psf=None, 
         fn_img_hr=None, suptitle=None, 
         fnfigout='test.pdf', vm=75, nbit=8):

    if fn_img.endswith('npy'):
        datalr = np.load(fn_img)[:, :]
    elif fn_img.endswith('png'):
      try:
          datalr = load_image(fn_img)
      except:
          datalr = load_image('demo/0851x4-crop.png')
    else:
      print('Do not recognize input image file type, exiting')
      exit()

#    datalr = hr2lr.normalize_data(datalr, nbit=nbit)

    if fn_img_hr!=None:
        if fn_img_hr.endswith('.npy'):
            datahr = np.load(fn_img_hr)
        elif fn_img_hr.endswith('png'):
            datahr = load_image(fn_img_hr)
    else:
        datahr = None

    if psf is not None:
        if datahr is None:
          pass
        print("Convolving data")
        if psf in ('gaussian','Gaussian'):
          kernel1D = signal.gaussian(8, std=1).reshape(8, 1)
          kernel = np.outer(kernel1D, kernel1D)
        elif psf.endswith('.npy'):
          kernel = np.load(psf)
          nkern = len(kernel)
          kernel = kernel[nkern//2-256:nkern//2+256, nkern//2-256:nkern//2+256]
        else:
          print("Can't interpret kernel")
          exit()
        plt.figure()
        plt.subplot(121)
        plt.imshow(datahr, vmax=25, vmin=5)
        datalr = hr2lr.convolvehr(datahr, kernel, rebin=1)
        datahr = hr2lr.normalize_data(datahr, nbit=nbit)
        plt.subplot(122)
        print(datalr.shape)
        datalr = hr2lr.normalize_data(datalr, nbit=nbit)
        plt.imshow(datalr, vmax=50, vmin=20)
        plt.show()
    else:
        print("Assuming data is already convolved")

    model = wdsr_b(scale=4, num_res_blocks=32)
    model.load_weights(fn_model)
    datalr = datalr[:,:,None]
    if nbit==8:
      datasr = resolve_single(model, datalr)
    else:
      datasr = resolve_single16(model, datalr)
    plotter(datalr, datasr, datahr=datahr, 
            suptitle=suptitle, fnfigout=fnfigout, vm=vm, 
            nbit=nbit)

if __name__=='__main__':
    # Example usage:
    # Generate images on training data:
    # for im in ./images/PSF-nkern64-4x/train/X4/*png;do python generate-hr.py $im ./weights-psf-4x.h5;done
    # Generate images on validation data
    # for im in ./images/PSF-nkern64-4x/valid/*png;do python generate-hr.py $im ./weights-psf-4x.h5;done

    parser = optparse.OptionParser(prog="hr2lr.py",
                                   version="",
                                   usage="%prog image weights.h5  [OPTIONS]",
                                   description="Take high resolution images, convolve them, \
                                   and save output.")

    parser.add_option('-f', dest='fnhr', 
                      help="high-res file name", default=None)
    parser.add_option('-k', '--psf', dest='psf', type='str',
                      help="If None, assume image is already low res", default=None)
    parser.add_option("-s", "--ksize", dest='ksize', type=int,
                      help="size of kernel", default=64)
    parser.add_option('-t', '--title', dest='title', type='str',
                      help="Super title for plot", default=None)
    parser.add_option('-o', '--fnfigout', dest='fnfigout', default='test.pdf')    
    parser.add_option('--sky', dest='sky', action="store_true",
                      help="use SKA mid image as input")
    parser.add_option('-r', '--rebin', dest='rebin', type=int,
                      help="factor to spatially rebin", default=4)
    parser.add_option('-b', '--nbit', dest='nbit', type=int,
                      help="number of bits in image", default=8)
    parser.add_option('--vm', dest='vm', type=int,
                      help="factor to spatially rebin", default=75)

    options, args = parser.parse_args()
    fn_img, fn_model = args
    
    func(fn_img, fn_model, psf=options.psf, 
         fn_img_hr=options.fnhr, 
         suptitle=options.title, 
         fnfigout=options.fnfigout,
         vm=options.vm, nbit=options.nbit)





