# GENERAL POINT:
# try reversing this, so try to capture everything you need to make a model that can produce
# the images that you'd expect to take
# this means that you can then build a max-liklihood model for your observation
# go through and look at RHL's slides from DESC PCWG


import numpy as np
import lsst.afw.image as afwImage


class Spectrum():
    """A (wavelength calibrated)"""
    # I don't think this is too heavyweight and could be useful over an ndarray
    spectrum2d = afwImage.maskedImage()

    # Probably go masked image as we need three planes
    # spectrum would be a class containing separate 1d spectra and a covariance matrix
    # tri-di or quin-di diagonal matrix? for now imagine a covariance matrix (don't care about memory)
    spectrum1d = None

    # psf as function of position (LSST PSF object? use PIFF? Does need to be 2d)
    # for optimal extraction do row-by-row using marginal version of PSF (taking profile of PSF - same iff Gaussian)
    # but we will need more so can say what it looks like at any point on the chip
    # can create marginal version of itself in any direction at any point (maybe only need to implement 0 and 90 deg for now)
    # zeroth order PSF (and therefore LSF) changes as you move the source, need to have hooks to handle this
    psf = None

    #### instrument model
    # ignoring the PSF, describe morphology along the dispersion axis
    # this is fully desceibed by the optical distortion of the camera
    # if you image a monochromatic deltra function, how and where does it appear (a PSF that looks like what, and where?)
    # (if you weren't dispersing this would just be the wcs)
    # if we always put star down in the same place this is simple
    # this is a function of temperature, as temp chagnes in the mirror effectively dilate the system
    # want a model that describes this
    # the dispersion solution here is _probably_ the same as for the astrometry here, and that could be useful
    # (so we measure on the sky)
    # defocusing to see how things move, chaning the temp etc, to build up this model
    # for this we need the throughput of the telescope
    spectroGeom = None

    # initially None as we haven't extracted/calibrated yet
    wavelengthSolution = None
    # will this be a 1d ndarray or 1d afwImage.maskedImage()
    # the afwImage could be useful for both API stuff, and holding a pixel mask
    # this is a Legendre/Chebyshev model of the residuals when we fit the instrument model which is the nominal wavelength solution

    # some description of the method used. Might just be a config object
    extractionMethod = None

    # no arclamp for auxTel?!
    # need some way of getting the geomtric solution


    # flag set to True once the calibration is performed
    isCalibrated = False

    # Some measurement of the residuals after subtracting fit
    # Probably wants renaming as it's probably not really a chi^2
    chiSq = None

    skyBackground = None
    skyBackgroundExtractionConfig = None
    # skySpectrum = some assumption?!

    # the spectrum of the star if it were observed above the atmosphere
    nominalSpectrum = None

    # some *rough* measurements of equivalent widths, useful for debugging
    # dict keyed by line-name as string, e.g. "H20"
    equivalentWidths = {}
    def calculateEquivalentWidth(line):
        return width

    def calibrate(self):
        # really this is probably done by a task, which creates the object
        wl = WavelengthSolution()
        wl.solve()
        self.isCalibrated = True
        self.wavelengthSolution = wl


    def getWavelengthAccuracyAtPixel():
        return

    def convolveWithKernel():
        return

    def resample():
        return

    def getColumnwiseFit():
        return columnSlice, fit, chiSq


class WavelengthSolution():
    # assume self-calibrating from lines as we have no arc lamp
    # starts from the geometric solution and refines from there
    # how stable is this? Can we use the nominal solution to find the Balmer series
    # and then solve using that

    # need to fit a continuum first and then fit lines on top of that

    orderBlockingFilters = []
    gratingType = ['RONCHI400', 'HOLOGRAPHIC_MADNESS', 'etc']
    directImageSubtracted = [True, False]
    visitInfo = "expTime, RA/Dec, objectId of source etc"

    # position of the source centroid on the CCD as floats
    # we need to know where it originally hit the detector
    # so that we can redo flatfielding
    sourceLocation = [x, y]

    def solve(self):
        """Find the wavelength solution"""
        return self

    def getWavelengthAtPixel(self, x, y):
        return wavelength

    def getPixelForWavelength(self, x, y):
        return (x, y)
