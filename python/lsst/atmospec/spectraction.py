#
# LSST Data Management System
#
# Copyright 2008-2018  AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope hat it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#

import os
import numpy as np

from spectractor import parameters
from spectractor.config import load_config
from spectractor.extractor.images import Image, find_target, turn_image
from spectractor.extractor.spectrum import (Spectrum, extract_spectrum_from_image, calibrate_spectrum,
                                            calibrate_spectrum_with_lines)
from spectractor.extractor.dispersers import Hologram

import lsst.log as lsstLog


class SpectractorShim():
    """Class for running the Spectractor code.

    This is designed to provide an implementation of the top-level function in
    Spectractor.spectractor.extractor.extractor.Spectractor()."""
    TRANSPOSE = True

    # leading * for kwargs only in constructor
    def __init__(self, *, configFile=None, paramOverrides=None, supplementaryParameters=None):
        if configFile:
            print(f"Loading config from {configFile}")
            load_config(configFile)
        self.log = lsstLog.getLogger(__name__)
        if paramOverrides is not None:
            self.overrideParameters(paramOverrides)
        if supplementaryParameters is not None:
            self.supplementParameters(supplementaryParameters)
        return

    def overrideParameters(self, overrides):
        """Dict of Spectractor parameters to override.

        Default values are set in spectractor.parameters.py for use as consts.
        This method provides a means for overriding the parameters as needed.

        Parameters
        ----------
            overrides : `dict`
        Dict of overrides to apply. Warning is logged if keys are found that do
        not map to existing Spectractor parameters.
        """
        for k, v in overrides.items():
            # NB do not use hasattr(parameters, k) here, as this is broken by
            # the overloading of __getattr__ in parameters
            if k in dir(parameters):
                setattr(parameters, k, v)
            else:
                self.log.warn("Did not find attribute %s in parameters" % k)
                raise RuntimeError(f"{k} not set to {v} {self.dumpParameters()}")

    def supplementParameters(self, supplementaryItems):
        """Dict of Spectractor parameters to add to the parameters.

        Use this method to add entries to the parameter namespace that do not
        already exist.

        Parameters
        ----------
            supplementaryItems : `dict`
        Dict of parameters to add. Warning is logged if keys already exist,
        as these should be overridden rather than supplemented.
        """
        # NB avoid using the variable name `parameters` in this method
        # due to scope collision
        for k, v in supplementaryItems.items():
            # NB do not use hasattr(parameters, k) here, as this is broken by
            # the overloading of __getattr__ in parameters
            if k in dir(parameters):
                msg = ("Supplementary parameter already existed %s in parameters,"
                       " use overrideParameters() to override already existing keys instead.")
                self.log.warn(msg, k)
            else:
                setattr(parameters, k, v)

    @staticmethod
    def dumpParameters():
        for item in dir(parameters):
            if not item.startswith("__"):
                print(item, getattr(parameters, item))

    def spectractorImageFromLsstExposure(self, exp, target_label='', disperser_label=''):
        """Construct a Spectractor Image object from LSST objects.

        Internally we try to use functions that calculate things and return
        them and set the values using the return rather than modifying the
        object in place where possible. Where this is not possible the methods
        are labeled _setSomething().
        """
        file_name = '/home/mfl/lsst/Spectractor/tests/data/asdasauxtel_first_light-1.fits'  # xxx REALLY needs removing
        image = Image(file_name=file_name, target_label=target_label, disperser_label=disperser_label)

        image.data = self._getImageData(exp)
        self._setReadNoiseFromExp(image, exp, 1)
        # xxx remove hard coding of 1 below!
        image.gain = self._setGainFromExp(image, exp, .85)  # gain required for calculating stat err
        self._setStatErrorInImage(image, exp, useExpVariance=False)
        # image.coord as an astropy SkyCoord - currently unused

        self._setImageAndHeaderInfo(image, exp)  # sets image attributes

        # image.expo = exp.getInfo().getVisitInfo().getExposureTime()  # set in header setter
        assert image.expo is not None
        assert image.expo != 0
        assert image.expo > 0

        image.convert_to_ADU_rate_units()  # divides by expTime and sets units to "ADU/s"

        image.disperser = Hologram(disperser_label, D=parameters.DISTANCE2CCD,
                                   data_dir=parameters.DISPERSER_DIR, verbose=parameters.VERBOSE)
        return image

    @staticmethod
    def _getFilterAndDisperserFromExp(exp):
        filterName = exp.getFilter().getName()
        if len(filterName.split('~')) == 1:
            filt1 = filterName
            filt2 = exp.getInfo().getMetadata()['GRATING']
        else:
            filt1, filt2 = filterName.split('~')  # TODO: import the delimiter from obs_package

        return filt1, filt2

    def _setImageAndHeaderInfo(self, image, exp, useVisitInfo=False):
        # currently set in spectractor.tools.extract_info_from_CTIO_header()
        filt, disperser = self._getFilterAndDisperserFromExp(exp)

        image.header.filter = filt
        image.header.disperser_label = disperser

        # exp time must be set in both header and in object attribute
        image.header.expo = exp.getInfo().getVisitInfo().getExposureTime()
        image.expo = exp.getInfo().getVisitInfo().getExposureTime()

        image.header['LSHIFT'] = 0.  # check if necessary
        image.header['D2CCD'] = parameters.DISTANCE2CCD  # necessary MFL

        try:
            if useVisitInfo:
                vi = exp.getInfo().getVisitInfo()
                image.header.airmass = vi.getBoresightAirmass()  # currently returns nan for obs_ctio0m9

            else:
                md = exp.getMetadata().toDict()
                image.header.airmass = md['AIRMASS']
                image.date_obs = md['DATE']
        except Exception:
            self.log.warn("Failed to set AIRMASS, default value of 1 used")
            image.header.airmass = 1.

        return

    def _getImageData(self, exp):
        if self.TRANSPOSE:
            return exp.maskedImage.image.array.T[:, ::-1]
        return exp.maskedImage.image.array

    def _setReadNoiseFromExp(self, spectractorImage, exp, constValue=None):
        # xxx need to implement this properly
        if constValue is not None:
            spectractorImage.read_out_noise = np.ones_like(spectractorImage.data) * constValue
        else:
            raise NotImplementedError("Setting noise image from exp variance not implemented")
            # spectractorImage.read_out_noise = np.ones_like(spectractorImage.data)
            # spectractorImage.read_out_noise = exp.variance.array

    def _setReadNoiseToNone(self, spectractorImage):
        spectractorImage.read_out_noise = None

    def _setStatErrorInImage(self, image, exp, useExpVariance=False):
        if useExpVariance:
            image.stat_errors = exp.maskedImage.variance.array  # xxx need to deal with TRANSPOSE here
        else:
            image.compute_statistical_error()

    def _setGainFromExp(self, spectractorImage, exp, constValue=None):
        # xxx need to implement this properly
        # Note that this is array-like and per-amplifier
        # so could use the code from gain flats
        if constValue:
            return np.ones_like(spectractorImage.data) * constValue
        return np.ones_like(spectractorImage.data)

    def _makePath(self, dirname, plotting=True):
        if plotting:
            dirname = os.path.join(dirname, 'plots')
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def _ensureFitsHeader(self, obj, dataDict=None):
        if 'SIMPLE' not in obj.header:
            obj.header.insert(0, ('SIMPLE', True))
        # if dataDict:
        #     header = obj.header
        #     for k, v in dataDict.items():
        #         if k not in header:
        #             header[k] = v

    @staticmethod
    def flipImageLeftRight(image, xpos, ypos):
        image.data = np.flip(image.data, 1)
        xpos = image.data.shape[1] - xpos
        return image, xpos, ypos

    @staticmethod
    def transposeCentroid(dmXpos, dmYpos, image):
        xSize, ySize = image.data.shape

        newX = dmXpos
        newY = -(dmYpos - ySize)
        return newY, newX

    def displayImage(self, image, centroid=None):
        import lsst.afw.image as afwImage
        import lsst.afw.display as afwDisp
        disp1 = afwDisp.Display(987, open=True)

        tempImg = afwImage.ImageF(np.zeros(image.data.shape, dtype=np.float32))
        tempImg.array[:] = image.data

        disp1.mtv(tempImg)
        if centroid:
            disp1.dot('x', centroid[0], centroid[1], size=100)

    def run(self, exp, xpos, ypos, target, outputRoot, expId):

        # run option kwargs in the original code - do something with these
        atmospheric_lines = True
        line_detection = True

        self.log.info('Starting SPECTRACTOR')
        # xxx change plotting to an option?
        self._makePath(outputRoot, plotting=True)  # early in case this fails, as processing is slow

        # xxx if/when objects are returned, change these to butler.put()
        outputFilenameSpectrum = os.path.join(outputRoot, 'v'+str(expId)+'_spectrum.fits')
        outputFilenameSpectrogram = os.path.join(outputRoot, 'v'+str(expId)+'_spectrogram.fits')
        outputFilenamePsf = os.path.join(outputRoot, 'v'+str(expId)+'_table.csv')

        # Load config file

        filt, disperser = self._getFilterAndDisperserFromExp(exp)
        image = self.spectractorImageFromLsstExposure(exp, target_label=target, disperser_label=disperser)

        if parameters.DEBUG:
            image.plot_image(scale='log10', target_pixcoords=(xpos, ypos))

        if self.TRANSPOSE:
            xpos, ypos = self.transposeCentroid(xpos, ypos, image)

        if disperser == 'ronchi170lpmm':  # TODO: add something more robust as to whether to flip!
            image, xpos, ypos = self.flipImageLeftRight(image, xpos, ypos)
            self.displayImage(image, centroid=(xpos, ypos))

        # Find the exact target position in the raw cut image: several methods

        # image turning and target finding - use LSST code instead?
        # and if not, at least test how the rotation code compares
        # this part of Spectractor is certainly slow at the very least
        if True:  # change this to be an option, at least for testing vs LSST
            self.log.info('Search for the target in the image...')
            _ = find_target(image, (xpos, ypos))  # sets the image.target_pixcoords
            turn_image(image)  # creates the rotated data, and sets the image.target_pixcoords_rotated

            # Rotate the image: several methods
            # Find the exact target position in the rotated image:
            # several methods - but how are these controlled? MFL
            self.log.info('Search for the target in the rotated image...')
            _ = find_target(image, (xpos, ypos), rotated=True)
        else:
            # code path for if the image is pre-rotated by LSST code
            raise NotImplementedError

        # Create Spectrum object
        spectrum = Spectrum(image=image)
        # Subtract background and bad pixels
        extract_spectrum_from_image(image, spectrum, w=parameters.PIXWIDTH_SIGNAL,
                                    ws=(parameters.PIXDIST_BACKGROUND,
                                        parameters.PIXDIST_BACKGROUND+parameters.PIXWIDTH_BACKGROUND),
                                    right_edge=parameters.CCD_IMSIZE-200)
        spectrum.atmospheric_lines = atmospheric_lines
        # Calibrate the spectrum
        calibrate_spectrum(spectrum)
        if line_detection:
            self.log.info('Calibrating order %d spectrum...' % spectrum.order)
            calibrate_spectrum_with_lines(spectrum)
        else:
            spectrum.header['WARNINGS'] = 'No calibration procedure with spectral features.'

        # Save the spectrum
        self._ensureFitsHeader(spectrum)  # SIMPLE is missing by default

        spectrum.save_spectrum(outputFilenameSpectrum, overwrite=True)
        spectrum.save_spectrogram(outputFilenameSpectrogram, overwrite=True)
        # Plot the spectrum

        parameters.DISPLAY = True
        if parameters.VERBOSE and parameters.DISPLAY:
            spectrum.plot_spectrum(xlim=None)
        distance = spectrum.chromatic_psf.get_distance_along_dispersion_axis()
        spectrum.lambdas = np.interp(distance, spectrum.pixels, spectrum.lambdas)
        spectrum.chromatic_psf.table['lambdas'] = spectrum.lambdas
        spectrum.chromatic_psf.table.write(outputFilenamePsf, overwrite=True)
        return spectrum
