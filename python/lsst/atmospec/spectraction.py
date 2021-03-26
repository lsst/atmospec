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
import astropy.coordinates as asCoords
from astropy import units as u

from spectractor import parameters
parameters.CALLING_CODE = "LSST_DM"  # this must be set IMMEDIATELY to supress colored logs

from spectractor.config import load_config  # noqa: E402
from spectractor.extractor.images import Image, find_target, turn_image  # noqa: E402

from spectractor.extractor.dispersers import Hologram  # noqa: E402
from spectractor.extractor.extractor import (set_fast_mode, FullForwardModelFitWorkspace,
                                             plot_comparison_truth, run_ffm_minimisation,
                                             extract_spectrum_from_image)  # noqa: E402
from spectractor.extractor.spectrum import Spectrum, calibrate_spectrum  # noqa: E402

import lsst.log as lsstLog  # noqa: E402


class SpectractorShim():
    """Class for running the Spectractor code.

    This is designed to provide an implementation of the top-level function in
    Spectractor.spectractor.extractor.extractor.Spectractor()."""
    TRANSPOSE = True

    # leading * for kwargs only in constructor
    def __init__(self, *, configFile=None, paramOverrides=None, supplementaryParameters=None,
                 resetParameters=None):
        if configFile:
            print(f"Loading config from {configFile}")
            load_config(configFile)
        self.log = lsstLog.getLogger(__name__)
        if paramOverrides is not None:
            self.overrideParameters(paramOverrides)
        if supplementaryParameters is not None:
            self.supplementParameters(supplementaryParameters)
        if resetParameters is not None:
            self.resetParameters(resetParameters)
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

    def resetParameters(self, resetParameters):
        """Dict of Spectractor parameters reset in the namespace.

        Use this method assign parameters to the namespace whether they exist
        or not.

        Parameters
        ----------
        resetParameters : `dict`
            Dict of parameters to add.
        """
        # NB avoid using the variable name `parameters` in this method
        # due to scope collision
        for k, v in resetParameters.items():
            # NB do not use hasattr(parameters, k) here, as this is broken by
            # the overloading of __getattr__ in parameters
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
        file_name = '/home/mfl/lsst/Spectractor/tests/data/auxtel_first_light-1.fits'  # xxx REALLY needs removing
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
        # TODO: import the delimiter from obs_package
        filterName = exp.getFilter().getName()
        if len(filterName.split('~')) == 1:
            filt1 = filterName
            filt2 = exp.getInfo().getMetadata()['GRATING']
        else:
            filt1, filt2 = filterName.split('~')

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
        newY = ySize - dmYpos  # image is also flipped in Y
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

    def setAdrParameters(self, spectrum, exp):
        # adr_params = [dec, hour_angle, temperature, pressure, humidity, airmass]
        vi = exp.getInfo().getVisitInfo()

        raDec = vi.getBoresightRaDec()
        dec = raDec.getDec()
        dec = asCoords.Angle(dec.asDegrees(), unit=u.deg)

        hourAngle = vi.getBoresightHourAngle()
        hourAngle = asCoords.Angle(hourAngle.asDegrees(), unit=u.deg)

        weather = vi.getWeather()
        _temperature = weather.getAirTemperature()
        temperature = _temperature if not np.isnan(_temperature) else 10  # maybe average?
        _pressure = weather.getAirPressure()
        pressure = _pressure if not np.isnan(_pressure) else 732  # nominal for altitude?
        _humidity = weather.getHumidity()
        humidity = _humidity if not np.isnan(_humidity) else None  # not a required param so no default

        airmass = vi.getBoresightAirmass()
        spectrum.adr_params = [dec, hourAngle, temperature, pressure, humidity, airmass]

    def run(self, exp, xpos, ypos, target, outputRoot, expId, binning=1, plotting=True):
        # run option kwargs in the original code, seems to ~always be True
        atmospheric_lines = True

        self.log.info('Starting SPECTRACTOR')
        # TODO: rename _makePath _makeOutputPath
        self._makePath(outputRoot, plotting=plotting)  # early in case this fails, as processing is slow

        # TODO: once objects are returned, change these to be a butler.put() or other persistence
        # TODO: change to f-strings
        outputFilenameSpectrum = os.path.join(outputRoot, 'v'+str(expId)+'_spectrum.fits')
        outputFilenameSpectrogram = os.path.join(outputRoot, 'v'+str(expId)+'_spectrogram.fits')
        outputFilenamePsf = os.path.join(outputRoot, 'v'+str(expId)+'_table.csv')
        outputFilenameLines = os.path.join(outputRoot, 'v'+str(expId)+'_lines.csv')

        # Upstream loads config file here

        # TODO: passing exact centroids seems to be causing a serious and non-obvious problem
        # this needs fixing for several reasons, mostly because if we have a
        # known good centroid then we want to skip the refitting entirely
        xpos = int(np.round(xpos))
        ypos = int(np.round(ypos))

        filt, disperser = self._getFilterAndDisperserFromExp(exp)
        image = self.spectractorImageFromLsstExposure(exp, target_label=target, disperser_label=disperser)
        if self.TRANSPOSE:
            xpos, ypos = self.transposeCentroid(xpos, ypos, image)

        if parameters.DEBUG:
            image.plot_image(scale='log10', target_pixcoords=(xpos, ypos))
            self.log.info(f"Pixel value at centroid = {image.data[int(ypos), int(xpos)]}")

        # XXX this needs removing or at least dealing with to not always just run! ASAP XXX
        # if disperser == 'ronchi170lpmm':  # TODO: add something more robust as to whether to flip!
        #     image, xpos, ypos = self.flipImageLeftRight(image, xpos, ypos)
        #     self.displayImage(image, centroid=(xpos, ypos))

        # TODO: set image.target_guess = np.asarray((xpos, ypos)) I think - might be unnecessary

        # TODO: this needs to move up a level to processStar.py, and run as a new process
        # Use fast mode
        if binning > 1:
            image.target_guess = (xpos, ypos)
            image = set_fast_mode(image)
            if parameters.DEBUG:
                image.plot_image(scale='symlog', target_pixcoords=image.target_guess)

        # image turning and target finding - use LSST code instead?
        # and if not, at least test how the rotation code compares
        # this part of Spectractor is certainly slow at the very least
        if True:  # TODO: change this to be an option, at least for testing vs LSST
            self.log.info('Search for the target in the image...')
            _ = find_target(image, image.target_guess)  # sets the image.target_pixcoords
            turn_image(image)  # creates the rotated data, and sets the image.target_pixcoords_rotated

            # Rotate the image: several methods
            # Find the exact target position in the rotated image:
            # several methods - but how are these controlled? MFL
            self.log.info('Search for the target in the rotated image...')
            _ = find_target(image, image.target_guess, rotated=True, use_wcs=False)
        else:
            # code path for if the image is pre-rotated by LSST code
            raise NotImplementedError

        # Create Spectrum object
        spectrum = Spectrum(image=image)
        self.setAdrParameters(spectrum, exp)

        # Subtract background and bad pixels
        extract_spectrum_from_image(image, spectrum, signal_width=parameters.PIXWIDTH_SIGNAL,
                                    ws=(parameters.PIXDIST_BACKGROUND,
                                        parameters.PIXDIST_BACKGROUND + parameters.PIXWIDTH_BACKGROUND),
                                    right_edge=parameters.CCD_IMSIZE)  # MFL: this used to be CCD_IMSIZE-200
        spectrum.atmospheric_lines = atmospheric_lines
        # Calibrate the spectrum
        calibrate_spectrum(spectrum)

        # Full forward model extraction: add transverse ADR and order 2 subtraction
        workspace = None
        if parameters.PSF_EXTRACTION_MODE == "PSF_2D":
            workspace = FullForwardModelFitWorkspace(spectrum, verbose=1, plot=True, live_fit=False,
                                                     amplitude_priors_method="spectrum")
            for i in range(2):
                spectrum.convert_from_flam_to_ADUrate()
                spectrum = run_ffm_minimisation(workspace, method="newton")

                # Calibrate the spectrum
                calibrate_spectrum(spectrum)
                workspace.p[1] = spectrum.disperser.D
                workspace.p[2] = spectrum.header['PIXSHIFT']
                # Compare with truth if available
                if parameters.PSF_EXTRACTION_MODE == "PSF_2D" and 'LBDAS_T' in spectrum.header and parameters.DEBUG:
                    plot_comparison_truth(spectrum, workspace)

        # Save the spectrum
        self._ensureFitsHeader(spectrum)  # SIMPLE is missing by default

        spectrum.save_spectrum(outputFilenameSpectrum, overwrite=True)
        spectrum.save_spectrogram(outputFilenameSpectrogram, overwrite=True)
        spectrum.lines.print_detected_lines(output_file_name=outputFilenameLines,
                                            overwrite=True, amplitude_units=spectrum.units)
        # Plot the spectrum

        parameters.DISPLAY = True
        if parameters.VERBOSE and parameters.DISPLAY:
            spectrum.plot_spectrum(xlim=None)

        # distance = spectrum.chromatic_psf.get_distance_along_dispersion_axis()
        # spectrum.lambdas = np.interp(distance, spectrum.pixels, spectrum.lambdas)
        spectrum.chromatic_psf.table['lambdas'] = spectrum.lambdas
        spectrum.chromatic_psf.table.write(outputFilenamePsf, overwrite=True)

        result = Spectraction()
        result.spectrum = spectrum
        result.image = image
        result.workspace = workspace
        result.lsstExp = exp

        return result  # XXX technically this should be a pipeBase.Struct I think


class Spectraction:
    """Simple class for holding any and all Spectractor outputs."""
    # TODO: make this a data class? Annoying with so many args being required
    # and positional though.

    # spectrum: Spectrum
    # spectrogram:
    # disperser: Hologram
    # image: Image
    # workspace: FullForwardModelFitWorkspace
