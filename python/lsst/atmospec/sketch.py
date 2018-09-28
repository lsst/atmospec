import numpy as np
import lsst.afw.image as afwImage


class Spectrum():
    """A (wavelength calibrated)"""
    # I don't think this is too heavyweight and could be useful over an ndarray
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

    def __init__(self, spectrum2d):
        # looks like np.zeros(1000, 100, dtype=float)
        self.spectrum2d = spectrum2d

    def calibrate(self):
        wl = WavelengthSolution()
        wl.solve()
        self.isCalibrated = True
        self.wavelengthSolution = wl


class WavelengthSolution():
    # position of the source centroid on the CCD as floats
    sourceLocation = [x, y]

    directImageSubtracted = [True, False]
    visitInfo = "expTime, RA/Dec, objectId of source etc"

    def solve(self):
        """Find the wavelength solution"""
        return self

    def getWavelengthAtPixel(self, x, y):
        wavelength = 0
        return wavelength

    def getPixelForWavelength(self, x, y):
        return (x, y)
