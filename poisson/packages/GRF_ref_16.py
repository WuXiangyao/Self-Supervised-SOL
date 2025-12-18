# Copyright 2017 Bruno Sciolla. All Rights Reserved.
# ==============================================================================
# Generator for 2D scale-invariant Gaussian Random Fields
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Main dependencies
import numpy
import scipy.fftpack


def fftind(size):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.

        Input args:
            size (integer): The size of the coordinate array to create
        Returns:
            k_ind, numpy array of shape (2, size, size) with:
                k_ind[0,:,:]:  k_x components
                k_ind[1,:,:]:  k_y components

        Example:

            print(fftind(5))

            [[[ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]]

            [[ 0  0  0  0  0]
            [ 1  1  1  1  1]
            [-3 -3 -3 -3 -3]
            [-2 -2 -2 -2 -2]
            [-1 -1 -1 -1 -1]]]

        """
    k_ind = numpy.mgrid[:size, :size] - int((size + 1) / 2)
    k_ind = scipy.fftpack.fftshift(k_ind)
    return (k_ind)

def gaussian_random_field(alpha=3.0,
                          size=128,
                          flag_normalize=True,
                          dtype2float=True):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.

        Input args:
            alpha (double, default = 3.0):
                The power of the power-law momentum distribution
            size (integer, default = 128):
                The size of the square output Gaussian Random Fields
            flag_normalize (boolean, default = True):
                Normalizes the Gaussian Field:
                    - to have an average of 0.0
                    - to have a standard deviation of 1.0

        Returns:
            gfield (numpy array of shape (size, size)):
                The random gaussian random field

        Example:
        import matplotlib
        import matplotlib.pyplot as plt
        example = gaussian_random_field()
        plt.imshow(example)
        """

    # Defines momentum indices
    k_idx = fftind(size)
    k_idx = numpy.mgrid[:size, :size]

    # Defines the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = numpy.power(k_idx[0] ** 2 + k_idx[1] ** 2 + 1e-10, -alpha / 2.0)
    amplitude[0, 0] = 0

    # Draws a complex gaussian random noise with normal
    # (circular) distributionq
    noise = numpy.random.normal(size=(size, size)) \
            + 1j * numpy.random.normal(size=(size, size))

    # To real space
    gfield = numpy.fft.ifft2(noise * amplitude).real
    # ################ Neumann
    # import fourierpack as sp
    # import chebypack as ch
    # import functools
    # iDCT = functools.partial(ch.Wrapper, [sp.icos_transform], dim=[-1, -2])
    # gfield = iDCT(torch.from_numpy(noise * amplitude)) / size / size
    # gfield = gfield.numpy()
    # ##########################


    # Sets the standard deviation to one
    if flag_normalize:
        gfield = gfield - numpy.mean(gfield)
        gfield = gfield / numpy.std(gfield)

    if dtype2float:
        gfield = gfield.astype(numpy.float32)
    
    return gfield

def gaussian_random_field_batch(batch,
                          alpha=3.0,
                          size=128,
                          flag_normalize=True):
    f = numpy.zeros([batch,size,size])
    for i in range(batch):
        f[i,:,:]=gaussian_random_field(alpha, size, flag_normalize=flag_normalize, dtype2float=False)

    return f

def main():
    import matplotlib
    import matplotlib.pyplot as plt
    example = gaussian_random_field(alpha=5)
    plt.imshow(example)
    plt.show()


if __name__ == '__main__':
    main()