# This file is part of atmospec.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import os
import numpy as np
import astropy.coordinates as asCoords
from astropy import units as u

from spectractor import parameters
parameters.CALLING_CODE = "LSST_DM"  # this must be set IMMEDIATELY to supress colored logs

from spectractor.config import load_config, apply_rebinning_to_parameters  # noqa: E402
from spectractor.extractor.images import Image, find_target, turn_image  # noqa: E402

from spectractor.extractor.dispersers import Hologram  # noqa: E402
from spectractor.extractor.extractor import (FullForwardModelFitWorkspace,  # noqa: E402
                                             run_ffm_minimisation,  # noqa: E402
                                             extract_spectrum_from_image,
                                             dumpParameters,
                                             run_spectrogram_deconvolution_psf2d)
from spectractor.extractor.spectrum import Spectrum, calibrate_spectrum  # noqa: E402
from spectractor.fit.fit_spectrum import SpectrumFitWorkspace, run_spectrum_minimisation  # noqa: E402
from spectractor.fit.fit_spectrogram import (SpectrogramFitWorkspace,  # noqa: E402
                                             run_spectrogram_minimisation)

from lsst.daf.base import DateTime  # noqa: E402
from .utils import getFilterAndDisperserFromExp  # noqa: E402


class SpectractorShim:
    """Class for running the Spectractor code.

    This is designed to provide an implementation of the top-level function in
    Spectractor.spectractor.extractor.extractor.Spectractor()."""

    # leading * for kwargs only in constructor
    def __init__(self, *, configFile=None, paramOverrides=None, supplementaryParameters=None,
                 resetParameters=None):
        if configFile:
            print(f"Loading config from {configFile}")
            load_config(configFile, rebin=False)
        self.log = logging.getLogger(__name__)
        if paramOverrides is not None:
            self.overrideParameters(paramOverrides)
        if supplementaryParameters is not None:
            self.supplementParameters(supplementaryParameters)
        if resetParameters is not None:
            self.resetParameters(resetParameters)

        if parameters.DEBUG:
            self.log.debug('Parameters pre-rebinning:')
            dumpParameters()

        return

    def overrideParameters(self, overrides):
        """Dict of Spectractor parameters to override.

        Default values are set in spectractor.parameters.py for use as consts.
        This method provides a means for overriding the parameters as needed.

        Parameters
        ----------
        overrides : `dict`
            Dict of overrides to apply. Warning is logged if keys are found
            that do not map to existing Spectractor parameters.
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
        """Print all the values in Spectractor's parameters module."""
        for item in dir(parameters):
            if not item.startswith("__"):
                print(item, getattr(parameters, item))

    def debugPrintTargetCentroidValue(self, image):
        """Print the positions and values of the centroid for debug purposes.

        Parameters
        ----------
        image : `spectractor.extractor.images.Image`
            The image.
        """
        x, y = image.target_guess
        self.log.debug(f"Image shape = {image.data.shape}")
        self.log.debug(f"x, y = {x}, {y}")
        x = int(np.round(x))
        y = int(np.round(y))
        self.log.debug(f"Value at {x}, {y} = {image.data[y, x]}")

    def spectractorImageFromLsstExposure(self, exp, xpos, ypos, *, target_label='',
                                         disperser_label='', filter_label=''):
        """Construct a Spectractor Image object from LSST objects.

        Internally we try to use functions that calculate things and return
        them and set the values using the return rather than modifying the
        object in place where possible. Where this is not possible the methods
        are labeled _setSomething().

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            The exposure to construct the image from.
        xpos : `float`
            The x position of the star's centroid.
        ypos : `float`
            The y position of the star's centroid.
        target_label : `str`, optional
            The name of the object, e.g. HD12345.
        disperser_label : `str`, optional
            The name of the dispersed, e.g. 'holo_003'
        filter_label : `str`, optional
            The name of the filter, e.g. 'SDSSi'

        Returns
        -------
        image : `spectractor.extractor.images.Image`
            The image.
        """
        # make a blank image, with the filter/disperser set
        image = Image(file_name='', target_label=target_label, disperser_label=disperser_label,
                      filter_label=filter_label)

        vi = exp.getInfo().getVisitInfo()
        rotAngle = vi.getBoresightRotAngle().asDegrees()
        parameters.OBS_CAMERA_ROTATION = 270 - (rotAngle % 360)

        radec = vi.getBoresightRaDec()
        image.ra = asCoords.Angle(radec.getRa().asDegrees(), unit="deg")
        image.dec = asCoords.Angle(radec.getDec().asDegrees(), unit="deg")
        ha = vi.getBoresightHourAngle().asDegrees()
        image.hour_angle = asCoords.Angle(ha, unit="deg")

        image.data = self._getImageData(exp)

        def _translateCentroid(dmXpos, dmYpos):
            # this function was necessary when we were sometimes transposing
            # and sometimes not. If we decide to always/never transpose then
            # this function can just be removed.
            newX = dmYpos
            newY = dmXpos
            return newX, newY
        image.target_guess = _translateCentroid(xpos, ypos)
        if parameters.DEBUG:
            self.debugPrintTargetCentroidValue(image)

        self._setReadNoiseFromExp(image, exp, 8.5)
        # xxx remove hard coding of gains below!
        image.gain = self._setGainFromExp(image, exp, 1./1.3)  # gain required for calculating stat err
        self._setStatErrorInImage(image, exp, useExpVariance=False)
        self._setMask(image, exp)
        image.flat = None

        self._setImageAndHeaderInfo(image, exp)  # sets image attributes

        assert image.expo is not None
        assert image.expo != 0
        assert image.expo > 0

        image.convert_to_ADU_rate_units()  # divides by expTime and sets units to "ADU/s"

        image.disperser = Hologram(disperser_label, D=parameters.DISTANCE2CCD,
                                   data_dir=parameters.DISPERSER_DIR, verbose=parameters.VERBOSE)

        image.compute_parallactic_angle()

        return image

    def _setImageAndHeaderInfo(self, image, exp, useVisitInfo=True):
        # currently set in spectractor.tools.extract_info_from_CTIO_header()
        filt, disperser = getFilterAndDisperserFromExp(exp)

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
                image.airmass = vi.getBoresightAirmass()  # currently returns nan for obs_ctio0m9
                # TODO: DM-33731 work out if this should be UTC or TAI.
                image.date_obs = vi.date.toString(DateTime.UTC)
            else:
                md = exp.getMetadata().toDict()
                image.header.airmass = md['AIRMASS']
                image.airmass = md['AIRMASS']
                image.date_obs = md['DATE']
        except Exception:
            self.log.warn("Failed to set AIRMASS, default value of 1 used")
            image.header.airmass = 1.

        # get supplementary metadata
        md = exp.getMetadata().toDict()
        image.header['DOMEAZ'] = md['DOMEAZ']
        image.header['AZ'] = md['AZSTART']
        image.header['EL'] = md['ELSTART']
        image.header['RA'] = md['RA']
        image.header['MJD'] = md['MJD']
        image.header['WINDSPD'] = md['WINDSPD']
        image.header['WINDDIR'] = md['WINDDIR']

        return

    def _getImageData(self, exp, trimToSquare=False):
        if trimToSquare:
            data = exp.image.array[0:4000, 0:4000]
        else:
            data = exp.image.array
        return self._transformArrayFromExpToImage(data)

    def _transformArrayFromExpToImage(self, array):
        # apply transformation on an exposure array to have correct orientation
        # in Spectractor Image
        return array.T[::, ::]

    def _setReadNoiseFromExp(self, spectractorImage, exp, constValue=None):
        # xxx need to implement this properly
        if constValue is not None:
            spectractorImage.read_out_noise = np.ones_like(spectractorImage.data) * constValue
        else:
            # TODO: Check with Jeremy if we want the raw read noise
            # or the per-pixel variance. Either is doable, just need to know.
            raise NotImplementedError("Setting noise image from exp variance not implemented")

    def _setReadNoiseToNone(self, spectractorImage):
        spectractorImage.read_out_noise = None

    def _setStatErrorInImage(self, image, exp, useExpVariance=False):
        if useExpVariance:
            image.err = self._transformArrayFromExpToImage(np.sqrt(exp.getVariance().array))
        else:
            image.compute_statistical_error()

    def _setMask(self, image, exp):
        badBit = exp.getMask().getPlaneBitMask("BAD")
        crBit = exp.getMask().getPlaneBitMask("CR")
        badPixels = np.logical_or((exp.getMask().array & badBit) > 0, (exp.getMask().array & crBit) > 0)
        image.mask = self._transformArrayFromExpToImage(badPixels)

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
        newX = dmYpos
        newY = xSize - dmXpos
        return newX, newY

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
        # The adr_params parameter format expected by spectractor are:
        # [dec, hour_angle, temperature, pressure, humidity, airmass]
        vi = exp.getInfo().getVisitInfo()

        raDec = vi.getBoresightRaDec()
        dec = raDec.getDec()
        dec = asCoords.Angle(dec.asDegrees(), unit=u.deg)

        hourAngle = vi.getBoresightHourAngle()
        hourAngle = asCoords.Angle(hourAngle.asDegrees(), unit=u.deg)

        weather = vi.getWeather()

        _temperature = weather.getAirTemperature()
        if _temperature is None or np.isnan(_temperature):
            self.log.warning("Temperature not set, using nominal value of 10 C")
            _temperature = 10  # nominal value
        temperature = _temperature

        _pressure = weather.getAirPressure()
        if _pressure is not None and not np.isnan(_pressure):
            if _pressure > 10_000:
                _pressure /= 100  # convert from Pa to hPa
        else:
            self.log.warning("Pressure not set, using nominal value of 743 hPa")
            _pressure = 743  # nominal for altitude?
        pressure = _pressure
        _humidity = weather.getHumidity()
        humidity = _humidity if not np.isnan(_humidity) else None  # not a required param so no default

        airmass = vi.getBoresightAirmass()
        spectrum.adr_params = [dec, hourAngle, temperature, pressure, humidity, airmass]
        spectrum.pressure = pressure
        spectrum.humidity = humidity
        spectrum.airmass = airmass
        spectrum.temperature = temperature

    def run(self, exp, xpos, ypos, target, doFitAtmosphere, doFitAtmosphereOnSpectrogram,
            outputRoot=None, plotting=True):
        # run option kwargs in the original code, seems to ~always be True
        atmospheric_lines = True

        self.log.info('Starting SPECTRACTOR')
        # TODO: rename _makePath _makeOutputPath
        if outputRoot is not None:  # TODO: remove post Gen3 transition
            self._makePath(outputRoot, plotting=plotting)  # early in case this fails, as processing is slow

        # Upstream loads config file here

        filter_label, disperser = getFilterAndDisperserFromExp(exp)
        image = self.spectractorImageFromLsstExposure(exp, xpos, ypos, target_label=target,
                                                      disperser_label=disperser,
                                                      filter_label=filter_label)

        if parameters.DEBUG:
            self.debugPrintTargetCentroidValue(image)
            title = 'Raw image with input target location'
            image.plot_image(scale='symlog', target_pixcoords=image.target_guess, title=title)
            self.log.info(f"Pixel value at centroid = {image.data[int(xpos), int(ypos)]}")

        # XXX this needs removing or at least dealing with to not always
        # just run! ASAP XXX
        # if disperser == 'ronchi170lpmm':
        # TODO: add something more robust as to whether to flip!
        #     image, xpos, ypos = self.flipImageLeftRight(image, xpos, ypos)
        #     self.displayImage(image, centroid=(xpos, ypos))

        if parameters.CCD_REBIN > 1:
            self.log.info(f'Rebinning image with rebin of {parameters.CCD_REBIN}')
            apply_rebinning_to_parameters()
            image.rebin()
            if parameters.DEBUG:
                self.log.info('Parameters post-rebinning:')
                dumpParameters()
                self.debugPrintTargetCentroidValue(image)
                title = 'Rebinned image with input target location'
                image.plot_image(scale='symlog', target_pixcoords=image.target_guess, title=title)
                self.log.debug('Post rebin:')
                self.debugPrintTargetCentroidValue(image)

        # image turning and target finding - use LSST code instead?
        # and if not, at least test how the rotation code compares
        # this part of Spectractor is certainly slow at the very least
        if True:  # TODO: change this to be an option, at least for testing vs LSST
            self.log.info('Search for the target in the image...')
            # sets the image.target_pixcoords
            _ = find_target(image, image.target_guess, widths=(parameters.XWINDOW, parameters.YWINDOW))
            turn_image(image)  # creates the rotated data, and sets the image.target_pixcoords_rotated

            # Rotate the image: several methods
            # Find the exact target position in the rotated image:
            # several methods - but how are these controlled? MFL
            self.log.info('Search for the target in the rotated image...')
            _ = find_target(image, image.target_guess, rotated=True,
                            widths=(parameters.XWINDOW_ROT, parameters.YWINDOW_ROT))
        else:
            # code path for if the image is pre-rotated by LSST code
            raise NotImplementedError

        # Create Spectrum object
        spectrum = Spectrum(image=image, order=parameters.SPEC_ORDER)
        self.setAdrParameters(spectrum, exp)

        # Subtract background and bad pixels
        w_psf1d, bgd_model_func = extract_spectrum_from_image(image, spectrum,
                                                              signal_width=parameters.PIXWIDTH_SIGNAL,
                                                              ws=[parameters.PIXDIST_BACKGROUND,
                                                                  parameters.PIXDIST_BACKGROUND
                                                                  + parameters.PIXWIDTH_BACKGROUND])
        spectrum.atmospheric_lines = atmospheric_lines
        if plotting:
            spectrum.plot_spectrum()

        # PSF2D deconvolution
        if parameters.SPECTRACTOR_DECONVOLUTION_PSF2D:
            run_spectrogram_deconvolution_psf2d(spectrum, bgd_model_func=bgd_model_func)

        # Calibrate the spectrum
        self.log.info(f'Calibrating order {spectrum.order:d} spectrum...')
        with_adr = True
        if parameters.OBS_OBJECT_TYPE != "STAR":
            # XXX Check what this is set to, and how
            # likely need to be passed through
            with_adr = False
        calibrate_spectrum(spectrum, with_adr=with_adr)

        # not necessarily set during fit but required to be present for astropy
        # fits writing to work (required to be in keeping with upstream)
        spectrum.data_next_order = np.zeros_like(spectrum.lambdas)
        spectrum.err_next_order = np.zeros_like(spectrum.lambdas)

        # Full forward model extraction:
        # adds transverse ADR and order 2 subtraction
        ffmWorkspace = None
        if parameters.SPECTRACTOR_DECONVOLUTION_FFM:
            ffmWorkspace = FullForwardModelFitWorkspace(spectrum, verbose=parameters.VERBOSE,
                                                        plot=True,
                                                        live_fit=False,
                                                        amplitude_priors_method="spectrum")
            spectrum = run_ffm_minimisation(ffmWorkspace, method="newton", niter=2)

        # Fit the atmosphere on the spectrum using uvspec binary
        spectrumAtmosphereWorkspace = None
        if doFitAtmosphere:
            spectrumAtmosphereWorkspace = SpectrumFitWorkspace(spectrum,
                                                               fit_angstrom_exponent=True,
                                                               verbose=parameters.VERBOSE,
                                                               plot=True)
            run_spectrum_minimisation(spectrumAtmosphereWorkspace, method="newton")

        # Fit the atmosphere directly on the spectrogram using uvspec binary
        spectrogramAtmosphereWorkspace = None
        if doFitAtmosphereOnSpectrogram:
            spectrogramAtmosphereWorkspace = SpectrogramFitWorkspace(spectrum,
                                                                     fit_angstrom_exponent=True,
                                                                     verbose=parameters.VERBOSE,
                                                                     plot=True)
            run_spectrogram_minimisation(spectrogramAtmosphereWorkspace, method="newton")

        # Save the spectrum
        self._ensureFitsHeader(spectrum)  # SIMPLE is missing by default

        # Plot the spectrum
        parameters.DISPLAY = True
        if parameters.VERBOSE and parameters.DISPLAY:
            spectrum.plot_spectrum(xlim=None)

        result = Spectraction()
        result.spectrum = spectrum
        result.image = image
        result.spectrumForwardModelFitParameters = ffmWorkspace.params if ffmWorkspace is not None else None
        result.spectrumLibradtranFitParameters = (spectrumAtmosphereWorkspace.params if
                                                  spectrumAtmosphereWorkspace is not None else None)
        result.spectrogramLibradtranFitParameters = (spectrogramAtmosphereWorkspace.params if
                                                     spectrogramAtmosphereWorkspace is not None else None)

        return result


class Spectraction:
    """A simple class for holding the Spectractor outputs.

    Will likely be updated in future to provide some simple getters to allow
    easier access to parts of the data structure, and perhaps some convenience
    methods for interacting with the more awkward objects (e.g. the Lines).
    """
    # result.spectrum = spectrum
    # result.image = image
    # result.w = w
