import numpy as np
import lsst.afw.image as afwImage


class Spectrum():
    """A (wavelength calibrated)"""
    # I don't think this is too heavyweight and could be useful over an ndarray

    def __init__(self, spectrum2d, metadata):
        # looks like np.zeros(1000, 100, dtype=float)
        self.spectrum2d = spectrum2d

    spectrum2d = afwImage.maskedImage()

    # initially None as we haven't extracted/calibrated yet
    spectrum1d = None

    # initially None as we haven't extracted/calibrated yet
    wavelengthSolution = None
    # will this be a 1d ndarray or 1d afwImage.maskedImage()
    # the afwImage could be useful for both API stuff, and holding a pixel mask

    # some description of the method used. Might just be a config object
    extractionMethod = None

    # flag set to True once the calibration is performed
    isCalibrated = False

    # Some measurement of the residuals after subtracting fit
    # Probably wants renaming as it's probably not really a chi^2
    chiSq = None

    lsfOrPsf = None

    skyBackground = None
    skyBackgroundExtractionConfig = None
    # skySpectrum = someAssumption?!

    # no arclamp for auxTel?!
    # need some way of getting the geomtric solution

    # def calibrate(self):
    # really this is probably done by a task, which creates the object
    #     wl = WavelengthSolution()
    #     wl.solve()
    #     self.isCalibrated = True
    #     self.wavelengthSolution = wl


    def getWavelengthAtPixel(self, x, y):
        return wavelength

    def getPixelForWavelength(self, x, y):
        return (x, y)
        
    def wavelengthAccuracyAtPixel():
        return
 
    def convolveWithKernel():
        return
        
    def resample():
        return
       
    def getColumnwiseFit():
        return columnSlice, fit, chiSq

    def calculateEquivalentWidth(line):
        return width



class WavelengthSolution():
    # assume self-calibrating from lines as we have no arc lamp
    # starts from the geometric solution and refines from there
    # how stable is this? Can we use the nominal solution to find the Balmer series
    # and then solve using that

    # need to fit a continuum first and then fit lines on top of that

    # position of the source centroid on the CCD as floats
    sourceLocation = [x, y]
    orderBlockingFilters = []
    gratingType = ['RONCHI400', 'HOLOGRAPHIC_MADNESS', 'etc']
    directImageSubtracted = [True, False]
    visitInfo = "expTime, RA/Dec, objectId of source etc"

    # we need to know where it originally hit the detector
    # so that we can redo flatfielding
    detectorLocation = [x, y]


    def solve(self):
        """Find the wavelength solution"""
        return self








