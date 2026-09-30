#!/usr/bin/env python
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
"""Test cases for the atmospec SpectractorShim."""

import importlib.resources
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import lsst.utils
import lsst.utils.tests
import lsst.afw.coord as afwCoord
import lsst.afw.image as afwImage
import lsst.geom as geom
from lsst.daf.base import DateTime

from spectractor import parameters
from spectractor.extractor.images import Image
import lsst.atmospec.spectraction as spectractionModule
from lsst.atmospec.spectraction import SpectractorShim, Spectraction


def _makeExposure(width=20, height=24, physicalFilter="empty~holo4_003",
                  weather=None, withMetadata=True):
    """Build a small ExposureF with a populated VisitInfo and filter."""
    exp = afwImage.ExposureF(width, height)
    exp.image.array[:] = np.random.rand(height, width).astype(np.float32) + 10.0
    exp.variance.array[:] = 1.0

    if weather is None:
        weather = afwCoord.Weather(10.0, 74300.0, 40.0)

    vi = afwImage.VisitInfo(
        exposureTime=30.0,
        date=DateTime("2021-01-01T00:00:00Z", DateTime.UTC),
        boresightRaDec=geom.SpherePoint(30.0 * geom.degrees, -30.0 * geom.degrees),
        boresightAzAlt=geom.SpherePoint(0.0 * geom.degrees, 80.0 * geom.degrees),
        boresightAirmass=1.1,
        boresightRotAngle=45.0 * geom.degrees,
        weather=weather,
    )
    exp.getInfo().setVisitInfo(vi)
    exp.setFilter(afwImage.FilterLabel(physical=physicalFilter, band="white"))
    if withMetadata:
        md = exp.getMetadata()
        md['GRATING'] = 'holo4_003'
        md['AIRMASS'] = 1.1
        md['DATE'] = '2021-01-01T00:00:00'
    return exp


def _makeSpectractorImage(shape=(10, 12)):
    """Make a real (blank) Spectractor Image usable by helper methods."""
    image = Image(file_name='', target_label='', disperser_label='', filter_label='')
    image.data = np.ones(shape)
    return image


class SpectractionStaticTestCase(lsst.utils.tests.TestCase):
    """Tests for the pure/static helper methods."""

    def testImport(self):
        import lsst.atmospec.spectraction as s  # noqa: F401

    def test_flipImageLeftRight(self):
        image = mock.MagicMock()
        data = np.arange(12).reshape(3, 4)
        image.data = data.copy()
        outImage, xpos, ypos = SpectractorShim.flipImageLeftRight(image, 1.0, 2.0)
        # data flipped along axis 1
        np.testing.assert_array_equal(outImage.data, np.flip(data, 1))
        self.assertEqual(xpos, data.shape[1] - 1.0)
        self.assertEqual(ypos, 2.0)

    def test_transposeCentroid(self):
        image = mock.MagicMock()
        image.data = np.ones((5, 7))  # xSize, ySize = 5, 7
        newX, newY = SpectractorShim.transposeCentroid(2.0, 3.0, image)
        self.assertEqual(newX, 3.0)  # dmYpos
        self.assertEqual(newY, 5 - 2.0)  # xSize - dmXpos

    def test_dumpParameters(self):
        # Should simply print without error.
        SpectractorShim.dumpParameters()


class SpectractionParameterTestCase(lsst.utils.tests.TestCase):
    """Tests for the parameter (over/supplement/reset) machinery."""

    def setUp(self):
        self.shim = SpectractorShim()

    def test_overrideParametersValid(self):
        original = parameters.VERBOSE
        try:
            self.shim.overrideParameters({'VERBOSE': True})
            self.assertTrue(parameters.VERBOSE)
        finally:
            parameters.VERBOSE = original

    def test_overrideParametersInvalidRaises(self):
        with self.assertRaises(RuntimeError):
            self.shim.overrideParameters({'THIS_PARAM_DOES_NOT_EXIST_XYZ': 1})

    def test_supplementParametersNew(self):
        name = 'MY_SUPPLEMENTARY_PARAM_XYZ'
        self.assertNotIn(name, dir(parameters))
        try:
            self.shim.supplementParameters({name: 42})
            self.assertEqual(getattr(parameters, name), 42)
        finally:
            if hasattr(parameters, name):
                delattr(parameters, name)

    def test_supplementParametersExistingWarns(self):
        # Existing key should NOT be modified, only a warning emitted.
        original = parameters.VERBOSE
        try:
            self.shim.supplementParameters({'VERBOSE': not original})
            self.assertEqual(parameters.VERBOSE, original)
        finally:
            parameters.VERBOSE = original

    def test_resetParameters(self):
        name = 'MY_RESET_PARAM_XYZ'
        try:
            self.shim.resetParameters({name: 7})
            self.assertEqual(getattr(parameters, name), 7)
            # reset also works for existing attributes
            original = parameters.VERBOSE
            self.shim.resetParameters({'VERBOSE': not original})
            self.assertEqual(parameters.VERBOSE, not original)
            parameters.VERBOSE = original
        finally:
            if hasattr(parameters, name):
                delattr(parameters, name)


class SpectractionInitTestCase(lsst.utils.tests.TestCase):
    """Tests for the SpectractorShim constructor."""

    @classmethod
    def setUpClass(cls):
        with importlib.resources.path("lsst.atmospec", "resources/config/auxtel.ini") as cfg:
            cls.configFile = str(cfg)

    def test_initDefault(self):
        shim = SpectractorShim()
        self.assertIsNotNone(shim.log)

    def test_initWithConfigFile(self):
        shim = SpectractorShim(configFile=self.configFile)
        self.assertIsNotNone(shim.log)

    def test_initWithParamDicts(self):
        supName = 'INIT_SUPP_PARAM_XYZ'
        resName = 'INIT_RESET_PARAM_XYZ'
        original = parameters.VERBOSE
        try:
            shim = SpectractorShim(
                paramOverrides={'VERBOSE': True},
                supplementaryParameters={supName: 1},
                resetParameters={resName: 2},
            )
            self.assertIsNotNone(shim.log)
            self.assertTrue(parameters.VERBOSE)
            self.assertEqual(getattr(parameters, supName), 1)
            self.assertEqual(getattr(parameters, resName), 2)
        finally:
            parameters.VERBOSE = original
            for name in (supName, resName):
                if hasattr(parameters, name):
                    delattr(parameters, name)

    def test_initWithDebug(self):
        original = parameters.DEBUG
        try:
            parameters.DEBUG = True
            shim = SpectractorShim()
            self.assertIsNotNone(shim.log)
        finally:
            parameters.DEBUG = original


class SpectractionHelperTestCase(lsst.utils.tests.TestCase):
    """Tests for the small IO / image helper methods."""

    def setUp(self):
        self.shim = SpectractorShim()

    def test_makePathPlotting(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, 'sub')
            self.shim._makePath(target, plotting=True)
            self.assertTrue(os.path.exists(os.path.join(target, 'plots')))

    def test_makePathNoPlotting(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, 'sub2')
            self.shim._makePath(target, plotting=False)
            self.assertTrue(os.path.exists(target))
            # Calling again when it already exists should be a no-op.
            self.shim._makePath(target, plotting=False)
            self.assertTrue(os.path.exists(target))

    def test_ensureFitsHeaderAdds(self):
        obj = _makeSpectractorImage()
        if 'SIMPLE' in obj.header:
            del obj.header['SIMPLE']
        self.shim._ensureFitsHeader(obj)
        self.assertIn('SIMPLE', obj.header)

    def test_ensureFitsHeaderAlreadyPresent(self):
        obj = _makeSpectractorImage()
        obj.header['SIMPLE'] = True
        self.shim._ensureFitsHeader(obj)
        self.assertIn('SIMPLE', obj.header)

    def test_getImageData(self):
        exp = _makeExposure()
        data = self.shim._getImageData(exp)
        # transposed relative to exp.image.array
        self.assertEqual(data.shape, exp.image.array.T.shape)

    def test_getImageDataTrimToSquare(self):
        exp = _makeExposure()
        data = self.shim._getImageData(exp, trimToSquare=True)
        # image is smaller than 4000, so trimming leaves it unchanged (then .T)
        self.assertEqual(data.shape, exp.image.array.T.shape)

    def test_setReadNoiseFromExpConst(self):
        image = _makeSpectractorImage()
        exp = _makeExposure()
        self.shim._setReadNoiseFromExp(image, exp, constValue=3.0)
        self.assertEqual(image.read_out_noise.shape, image.data.shape)
        self.assertTrue(np.all(image.read_out_noise == 3.0))

    def test_setReadNoiseFromExpNoneRaises(self):
        image = _makeSpectractorImage()
        exp = _makeExposure()
        with self.assertRaises(NotImplementedError):
            self.shim._setReadNoiseFromExp(image, exp, constValue=None)

    def test_setReadNoiseToNone(self):
        image = _makeSpectractorImage()
        self.shim._setReadNoiseToNone(image)
        self.assertIsNone(image.read_out_noise)

    def test_setGainFromExpConst(self):
        image = _makeSpectractorImage()
        exp = _makeExposure()
        gain = self.shim._setGainFromExp(image, exp, constValue=0.85)
        self.assertEqual(gain.shape, image.data.shape)
        self.assertTrue(np.all(gain == 0.85))

    def test_setGainFromExpDefault(self):
        image = _makeSpectractorImage()
        exp = _makeExposure()
        gain = self.shim._setGainFromExp(image, exp, constValue=None)
        self.assertTrue(np.all(gain == 1.0))

    def test_setStatErrorInImageComputed(self):
        image = _makeSpectractorImage()
        exp = _makeExposure()
        image.read_out_noise = np.ones_like(image.data)
        image.gain = np.ones_like(image.data)
        self.shim._setStatErrorInImage(image, exp, useExpVariance=False)
        self.assertIsNotNone(image.err)

    def test_setStatErrorInImageFromVariance(self):
        image = _makeSpectractorImage(shape=(24, 20))
        exp = _makeExposure()
        self.shim._setStatErrorInImage(image, exp, useExpVariance=True)
        np.testing.assert_array_equal(image.stat_errors, exp.maskedImage.variance.array)

    def test_debugPrintTargetCentroidValue(self):
        image = mock.MagicMock()
        image.target_guess = (2.0, 3.0)
        image.data = np.ones((10, 10))
        # Should run without error.
        self.shim.debugPrintTargetCentroidValue(image)

    def test_setImageAndHeaderInfoVisitInfo(self):
        image = _makeSpectractorImage()
        exp = _makeExposure()
        self.shim._setImageAndHeaderInfo(image, exp, useVisitInfo=True)
        self.assertEqual(image.expo, 30.0)
        self.assertEqual(image.header.filter, 'empty')

    def test_setImageAndHeaderInfoMetadata(self):
        image = _makeSpectractorImage()
        exp = _makeExposure()
        self.shim._setImageAndHeaderInfo(image, exp, useVisitInfo=False)
        self.assertEqual(image.airmass, 1.1)
        self.assertEqual(image.date_obs, '2021-01-01T00:00:00')

    def test_setImageAndHeaderInfoException(self):
        # Missing AIRMASS in the metadata triggers the exception branch,
        # falling back to a default airmass of 1.
        image = _makeSpectractorImage()
        exp = _makeExposure()
        exp.getMetadata().remove('AIRMASS')
        self.shim._setImageAndHeaderInfo(image, exp, useVisitInfo=False)
        self.assertEqual(image.header.airmass, 1.)

    def test_displayImage(self):
        image = mock.MagicMock()
        image.data = np.ones((5, 6))
        with mock.patch("lsst.afw.display.Display") as mockDisp:
            self.shim.displayImage(image, centroid=(1, 2))
            mockDisp.assert_called_once()
            self.assertTrue(mockDisp.return_value.mtv.called)
            self.assertTrue(mockDisp.return_value.dot.called)

    def test_displayImageNoCentroid(self):
        image = mock.MagicMock()
        image.data = np.ones((5, 6))
        with mock.patch("lsst.afw.display.Display") as mockDisp:
            self.shim.displayImage(image, centroid=None)
            self.assertFalse(mockDisp.return_value.dot.called)


class SpectractionAdrTestCase(lsst.utils.tests.TestCase):
    """Tests for setAdrParameters."""

    def setUp(self):
        self.shim = SpectractorShim()

    def test_setAdrParametersValidWeather(self):
        exp = _makeExposure(weather=afwCoord.Weather(10.0, 74300.0, 40.0))
        spectrum = mock.MagicMock()
        self.shim.setAdrParameters(spectrum, exp)
        # pressure converted Pa -> hPa
        self.assertAlmostEqual(spectrum.pressure, 743.0)
        self.assertEqual(spectrum.temperature, 10.0)
        self.assertEqual(spectrum.humidity, 40.0)
        self.assertEqual(len(spectrum.adr_params), 6)

    def test_setAdrParametersNanWeather(self):
        nan = float('nan')
        exp = _makeExposure(weather=afwCoord.Weather(nan, nan, nan))
        spectrum = mock.MagicMock()
        self.shim.setAdrParameters(spectrum, exp)
        # falls back to nominal values
        self.assertEqual(spectrum.temperature, 10)
        self.assertEqual(spectrum.pressure, 743)
        self.assertIsNone(spectrum.humidity)

    def test_setAdrParametersLowPressure(self):
        # pressure already in hPa (< 10000) should not be divided.
        exp = _makeExposure(weather=afwCoord.Weather(10.0, 743.0, 40.0))
        spectrum = mock.MagicMock()
        self.shim.setAdrParameters(spectrum, exp)
        self.assertAlmostEqual(spectrum.pressure, 743.0)


class SpectractionImageBuildTestCase(lsst.utils.tests.TestCase):
    """Tests for spectractorImageFromLsstExposure."""

    @classmethod
    def setUpClass(cls):
        with importlib.resources.path("lsst.atmospec", "resources/config/auxtel.ini") as cfg:
            cls.shim = SpectractorShim(configFile=str(cfg))

    def test_spectractorImageFromLsstExposure(self):
        exp = _makeExposure()
        with mock.patch.object(spectractionModule, "Hologram"), \
                mock.patch.object(Image, "compute_parallactic_angle"):
            image = self.shim.spectractorImageFromLsstExposure(
                exp, 5.0, 6.0, target_label='HD1',
                disperser_label='holo4_003', filter_label='empty')
        self.assertEqual(image.units, "ADU/s")
        self.assertEqual(image.expo, 30.0)
        # target_guess is the (y, x) translation of the input centroid
        self.assertEqual(tuple(image.target_guess), (6.0, 5.0))

    def test_spectractorImageFromLsstExposureDebug(self):
        exp = _makeExposure()
        original = parameters.DEBUG
        try:
            parameters.DEBUG = True
            with mock.patch.object(spectractionModule, "Hologram"), \
                    mock.patch.object(Image, "compute_parallactic_angle"):
                image = self.shim.spectractorImageFromLsstExposure(
                    exp, 5.0, 6.0, target_label='HD1',
                    disperser_label='holo4_003', filter_label='empty')
            self.assertIsNotNone(image)
        finally:
            parameters.DEBUG = original


class SpectractionRunTestCase(lsst.utils.tests.TestCase):
    """Tests for the run() pipeline, with heavy mocking of Spectractor."""

    @classmethod
    def setUpClass(cls):
        with importlib.resources.path("lsst.atmospec", "resources/config/auxtel.ini") as cfg:
            cls.shim = SpectractorShim(configFile=str(cfg))

    def setUp(self):
        self._saved = {name: getattr(parameters, name) for name in (
            'DEBUG', 'VERBOSE', 'CCD_REBIN', 'SPECTRACTOR_DECONVOLUTION_PSF2D',
            'SPECTRACTOR_DECONVOLUTION_FFM', 'OBS_OBJECT_TYPE', 'DISPLAY')}

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(parameters, name, value)

    def _runWithMocks(self, **runKwargs):
        exp = _makeExposure()
        spectrum = mock.MagicMock()
        spectrum.order = 1
        spectrum.lambdas = np.zeros(10)
        image = mock.MagicMock()
        image.data = np.ones((20, 24))
        image.target_guess = (5.0, 6.0)

        patches = {
            "spectractorImageFromLsstExposure": mock.patch.object(
                SpectractorShim, "spectractorImageFromLsstExposure", return_value=image),
            "setAdrParameters": mock.patch.object(SpectractorShim, "setAdrParameters"),
            "find_target": mock.patch.object(spectractionModule, "find_target"),
            "turn_image": mock.patch.object(spectractionModule, "turn_image"),
            "Spectrum": mock.patch.object(spectractionModule, "Spectrum", return_value=spectrum),
            "extract": mock.patch.object(
                spectractionModule, "extract_spectrum_from_image",
                return_value=(mock.MagicMock(), mock.MagicMock())),
            "psf2d": mock.patch.object(spectractionModule, "run_spectrogram_deconvolution_psf2d"),
            "calibrate": mock.patch.object(spectractionModule, "calibrate_spectrum"),
            "ffmWs": mock.patch.object(spectractionModule, "FullForwardModelFitWorkspace"),
            "ffmRun": mock.patch.object(
                spectractionModule, "run_ffm_minimisation", return_value=spectrum),
            "specWs": mock.patch.object(spectractionModule, "SpectrumFitWorkspace"),
            "specRun": mock.patch.object(spectractionModule, "run_spectrum_minimisation"),
            "sgramWs": mock.patch.object(spectractionModule, "SpectrogramFitWorkspace"),
            "sgramRun": mock.patch.object(spectractionModule, "run_spectrogram_minimisation"),
            "rebin": mock.patch.object(spectractionModule, "apply_rebinning_to_parameters"),
        }
        started = {k: p.start() for k, p in patches.items()}
        self.addCleanup(mock.patch.stopall)
        result = self.shim.run(exp, 5.0, 6.0, "HD1", **runKwargs)
        return result, spectrum, image, started

    def test_runAllBranchesOn(self):
        parameters.CCD_REBIN = 1
        parameters.VERBOSE = False
        parameters.DEBUG = False
        parameters.SPECTRACTOR_DECONVOLUTION_PSF2D = True
        parameters.SPECTRACTOR_DECONVOLUTION_FFM = True
        parameters.OBS_OBJECT_TYPE = "STAR"
        with tempfile.TemporaryDirectory() as tmp:
            result, spectrum, image, started = self._runWithMocks(
                doFitAtmosphere=True, doFitAtmosphereOnSpectrogram=True,
                outputRoot=tmp, plotting=True)
        self.assertIsInstance(result, Spectraction)
        self.assertIs(result.spectrum, spectrum)
        self.assertIs(result.image, image)
        self.assertIsNotNone(result.spectrumForwardModelFitParameters)
        self.assertIsNotNone(result.spectrumLibradtranFitParameters)
        self.assertIsNotNone(result.spectrogramLibradtranFitParameters)
        self.assertTrue(started["psf2d"].called)
        self.assertTrue(started["specRun"].called)
        self.assertTrue(started["sgramRun"].called)

    def test_runAllBranchesOff(self):
        parameters.CCD_REBIN = 1
        parameters.VERBOSE = False
        parameters.DEBUG = False
        parameters.SPECTRACTOR_DECONVOLUTION_PSF2D = False
        parameters.SPECTRACTOR_DECONVOLUTION_FFM = False
        parameters.OBS_OBJECT_TYPE = "HG-AR"  # triggers with_adr = False
        result, spectrum, image, started = self._runWithMocks(
            doFitAtmosphere=False, doFitAtmosphereOnSpectrogram=False,
            outputRoot=None, plotting=True)
        self.assertIsInstance(result, Spectraction)
        self.assertIsNone(result.spectrumForwardModelFitParameters)
        self.assertIsNone(result.spectrumLibradtranFitParameters)
        self.assertIsNone(result.spectrogramLibradtranFitParameters)
        self.assertFalse(started["psf2d"].called)
        self.assertFalse(started["ffmRun"].called)
        self.assertFalse(started["specRun"].called)
        self.assertFalse(started["sgramRun"].called)

    def test_runDebugAndRebin(self):
        parameters.CCD_REBIN = 2
        parameters.VERBOSE = True
        parameters.DEBUG = True
        parameters.SPECTRACTOR_DECONVOLUTION_PSF2D = True
        parameters.SPECTRACTOR_DECONVOLUTION_FFM = True
        parameters.OBS_OBJECT_TYPE = "STAR"
        result, spectrum, image, started = self._runWithMocks(
            doFitAtmosphere=True, doFitAtmosphereOnSpectrogram=True,
            outputRoot=None, plotting=True)
        self.assertIsInstance(result, Spectraction)
        # rebinning path was exercised
        self.assertTrue(started["rebin"].called)
        self.assertTrue(image.rebin.called)


class SpectractionContainerTestCase(lsst.utils.tests.TestCase):
    """Tests for the Spectraction data container."""

    def test_instantiate(self):
        result = Spectraction()
        result.spectrum = "spectrum"
        result.image = "image"
        self.assertEqual(result.spectrum, "spectrum")
        self.assertEqual(result.image, "image")


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
