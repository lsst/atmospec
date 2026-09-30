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
"""Test cases for lsst.atmospec.extraction."""

import types
import unittest
import unittest.mock as mock
import numpy as np

import lsst.utils
import lsst.utils.tests
import lsst.geom as geom
import lsst.afw.image as afwImage
from lsst.atmospec.dispersion import DispersionRelation
from lsst.atmospec.extraction import (
    SpectralExtractionTask,
    SpectralExtractionTaskConfig,
    gausMoffatModel,
    moffatModel,
)


def makeGaussTrace(width, height, amp=50.0, center=None, sigma=3.0, background=5.0):
    """Build a synthetic 2D array with a Gaussian spatial profile per row.

    Each row (constant y) contains a Gaussian in x, emulating a spectral
    trace whose spatial profile runs across the columns.
    """
    if center is None:
        center = width / 2.0
    xs = np.arange(width)
    profile = amp * np.exp(-(xs - center) ** 2 / (2.0 * sigma ** 2)) + background
    array = np.tile(profile, (height, 1)).astype(np.float32)
    return array


def makeExposure(width, height, **kwargs):
    """Make an ExposureF containing a synthetic Gaussian trace."""
    bbox = geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(width, height))
    exp = afwImage.ExposureF(bbox)
    exp.image.array[:] = makeGaussTrace(width, height, **kwargs)
    exp.variance.array[:] = 1.0
    return exp, bbox


class ModelFunctionTestCase(lsst.utils.tests.TestCase):
    """Tests for the module-level pure model/fit helpers."""

    def test_gauss1D(self):
        x = np.linspace(-10, 10, 101)
        amp, mu, sigma = 5.0, 0.0, 2.0
        y = SpectralExtractionTask.gauss1D(x, amp, mu, sigma)
        # Peak at mu, equal to amp.
        self.assertAlmostEqual(np.max(y), amp, places=6)
        self.assertAlmostEqual(x[np.argmax(y)], mu, places=6)
        # Symmetric about the mean.
        self.assertFloatsAlmostEqual(y, y[::-1], atol=1e-6)

    def test_moffatModel(self):
        model = moffatModel((10.0, 5.0, 2.0))
        self.assertAlmostEqual(model.amplitude.value, 10.0)
        self.assertAlmostEqual(model.x_0.value, 5.0)
        self.assertAlmostEqual(model.gamma.value, 2.0)
        # Peak of a Moffat is at x_0 and equals the amplitude.
        self.assertAlmostEqual(model(5.0), 10.0, places=6)

    def test_gausMoffatModel(self):
        pars = (10.0, 15.0, 1.5, 1.5, 500.0, 15.0, 3.0)
        model = gausMoffatModel(pars)
        # Compound model (Gaussian + Moffat) evaluates to a finite array.
        xs = np.arange(30)
        vals = model(xs)
        self.assertEqual(vals.shape, xs.shape)
        self.assertTrue(np.all(np.isfinite(vals)))
        # Sub-model parameters are accessible on the compound model.
        self.assertAlmostEqual(model.amplitude_0.value, 500.0)
        self.assertAlmostEqual(model.amplitude_1.value, 10.0)

    def test_moffatFit(self):
        pixels = np.arange(40)
        truth = moffatModel((1000.0, 20.0, 3.0))
        footprint = np.asarray(truth(pixels), dtype=float)
        integral, x0, gamma, alpha = SpectralExtractionTask.moffatFit(
            pixels, footprint, amp=900.0, mu=19.0, sigma=2.5)
        self.assertTrue(np.isfinite(integral))
        self.assertGreater(integral, 0.0)
        # The recovered centroid should be close to the injected one.
        self.assertAlmostEqual(x0, 20.0, delta=1.0)
        self.assertTrue(np.isfinite(gamma))
        self.assertTrue(np.isfinite(alpha))

    def test_gaussMoffatFit(self):
        pixels = np.arange(40)
        truthPars = (200.0, 20.0, 1.5, 1.5, 800.0, 20.0, 3.0)
        truth = gausMoffatModel(truthPars)
        footprint = np.asarray(truth(pixels), dtype=float)
        initialPars = (150.0, 20.0, 1.5, 1.5, 700.0, 20.0, 3.0)
        result = SpectralExtractionTask.gaussMoffatFit(pixels, footprint, initialPars)
        self.assertEqual(len(result), 9)
        intGM, intG, fitGausAmp, fitX0, fitGam, fitAlpha, fitMofAmp, fitMofMean, fitMofWid = result
        for val in result:
            self.assertTrue(np.isfinite(val))

    def test_subtractBkgd(self):
        width = 30
        bkgd_size = 5
        slc = np.zeros(width)
        slc[:] = 7.0  # constant background of 7
        slc[12:18] += 100.0  # a bump in the middle (inside the kept region)
        result = SpectralExtractionTask.subtractBkgd(slc, width, bkgd_size)
        # Kept region has length width - 2*bkgd_size.
        self.assertEqual(len(result), width - 2 * bkgd_size)
        # Constant background should be subtracted to ~zero at the edges of
        # the kept region.
        self.assertAlmostEqual(result[0], 0.0, places=6)


class ConfigTestCase(lsst.utils.tests.TestCase):
    """Tests for the task configuration."""

    def test_configDefaults(self):
        config = SpectralExtractionTaskConfig()
        self.assertFalse(config.perRowBackground)
        self.assertEqual(config.perRowBackgroundSize, 10)
        self.assertFalse(config.writeResiduals)
        self.assertTrue(config.doSmoothBackround)
        self.assertFalse(config.doSigmaClipBackground)
        config.validate()

    def test_configConstruction(self):
        task = SpectralExtractionTask()
        self.assertIsNotNone(task.config)
        task2 = SpectralExtractionTask(config=SpectralExtractionTaskConfig())
        self.assertIsNotNone(task2.config)


class BackgroundTestCase(lsst.utils.tests.TestCase):
    """Additional edge cases for _calculateBackground."""

    def test_noSmooth(self):
        task = SpectralExtractionTask()
        mi = afwImage.MaskedImageF(5, 20)
        mi.image.array[:] = 3.0
        bgImg = task._calculateBackground(mi, 5, smooth=False)
        self.assertEqual(np.shape(mi.image.array), np.shape(bgImg.array))
        self.assertFloatsAlmostEqual(np.max(bgImg.array), 3.0, atol=1e-5)

    def test_sigmaClip(self):
        config = SpectralExtractionTaskConfig()
        config.doSigmaClipBackground = True
        task = SpectralExtractionTask(config=config)
        mi = afwImage.MaskedImageF(5, 20)
        mi.image.array[:] = 2.0
        bgImg = task._calculateBackground(mi, 5)
        self.assertEqual(np.shape(mi.image.array), np.shape(bgImg.array))

    def test_reduceNbinsWarning(self):
        task = SpectralExtractionTask()
        mi = afwImage.MaskedImageF(5, 6)
        mi.image.array[:] = 1.0
        # nbins larger than height-1 triggers the reduction branch.
        bgImg = task._calculateBackground(mi, 50, smooth=False)
        self.assertEqual(np.shape(mi.image.array), np.shape(bgImg.array))


class InitialiseAndFluxTestCase(lsst.utils.tests.TestCase):
    """Tests for initialise() and getFluxBasic()."""

    def setUp(self):
        self.width = 30
        self.height = 25
        self.dispersion = DispersionRelation([100, 200], [400, 800])

    def _makeTask(self, config=None):
        return SpectralExtractionTask(config=config)

    def test_initialise(self):
        exp, bbox = makeExposure(self.width, self.height)
        task = self._makeTask()
        centroid = geom.Point2D(self.width / 2.0, self.height / 2.0)
        task.initialise(exp, centroid, bbox, self.dispersion)

        self.assertEqual(task.spectrumWidth, self.width)
        self.assertEqual(task.spectrumHeight, self.height)
        self.assertEqual(len(task.apertureFlux), self.height)
        self.assertEqual(len(task.psfFitPars), self.height)
        self.assertEqual(len(task.moffatFitPars), self.height)
        self.assertEqual(len(task.gausMoffatFitPars), self.height)
        # Background-subtracted image should exist and match the footprint.
        self.assertEqual(task.bgSubMi.getDimensions(), task.footprintMi.getDimensions())
        self.assertIs(task.dispersionRelation, self.dispersion)

    def test_initialise_noSmoothBackground(self):
        config = SpectralExtractionTaskConfig()
        config.doSmoothBackround = False
        exp, bbox = makeExposure(self.width, self.height)
        task = self._makeTask(config=config)
        centroid = geom.Point2D(self.width / 2.0, self.height / 2.0)
        task.initialise(exp, centroid, bbox, self.dispersion)
        self.assertEqual(task.spectrumHeight, self.height)

    def test_getFluxBasic(self):
        exp, bbox = makeExposure(self.width, self.height)
        task = self._makeTask()
        centroid = geom.Point2D(self.width / 2.0, self.height / 2.0)
        task.initialise(exp, centroid, bbox, self.dispersion)
        result = task.getFluxBasic()
        self.assertIs(result, task)

        # Aperture flux and row-wise max must be populated and finite.
        self.assertEqual(len(task.apertureFlux), self.height)
        self.assertTrue(np.all(np.isfinite(task.apertureFlux)))
        self.assertTrue(np.all(np.isfinite(task.rowWiseMax)))

        # Basic Gaussian fit should have succeeded on at least some rows,
        # recovering a centroid near the trace center.
        fittedRows = [p for p in task.psfFitPars if p is not None]
        self.assertGreater(len(fittedRows), 0)
        for (psfAmp, psfMu, psfSigma, psfFlux) in fittedRows:
            self.assertGreater(psfAmp, 0)
            self.assertGreater(psfFlux, 0)

    def test_getFluxBasic_perRowBackground(self):
        config = SpectralExtractionTaskConfig()
        config.perRowBackground = True
        config.perRowBackgroundSize = 10
        exp, bbox = makeExposure(self.width, self.height)
        task = self._makeTask(config=config)
        centroid = geom.Point2D(self.width / 2.0, self.height / 2.0)
        task.initialise(exp, centroid, bbox, self.dispersion)
        result = task.getFluxBasic()
        self.assertIs(result, task)
        self.assertEqual(len(task.apertureFlux), self.height)

    def test_getFluxBasic_offCenterWarning(self):
        # Trace centered near an edge so |mu - width/2| >= 10, triggering
        # the "initial mu more than 10 pixels from footprint center" warning.
        exp, bbox = makeExposure(self.width, self.height, center=3.0, sigma=2.0)
        task = self._makeTask()
        centroid = geom.Point2D(3.0, self.height / 2.0)
        task.initialise(exp, centroid, bbox, self.dispersion)
        result = task.getFluxBasic()
        self.assertIs(result, task)

    def test_getFluxBasic_writeResiduals(self):
        config = SpectralExtractionTaskConfig()
        config.writeResiduals = True
        exp, bbox = makeExposure(self.width, self.height)
        task = self._makeTask(config=config)
        centroid = geom.Point2D(self.width / 2.0, self.height / 2.0)
        task.initialise(exp, centroid, bbox, self.dispersion)
        result = task.getFluxBasic()
        self.assertIs(result, task)

    def test_getFluxBasic_preventRunaway(self):
        # Enable the PREVENT_RUNAWAY module flag and use an off-centre trace
        # with a large per-row value spread so that both the psfMu and the
        # psfSigma "runaway" reset branches fire.
        exp, bbox = makeExposure(self.width, self.height, amp=100.0,
                                 center=3.0, sigma=3.0)
        task = self._makeTask()
        centroid = geom.Point2D(3.0, self.height / 2.0)
        task.initialise(exp, centroid, bbox, self.dispersion)
        with mock.patch("lsst.atmospec.extraction.PREVENT_RUNAWAY", True):
            result = task.getFluxBasic()
        self.assertIs(result, task)

    def test_getFluxBasic_failingRows(self):
        # Row 0 has a clean Gaussian (so the bootstrap succeeds), but some
        # later rows are driven below background so their max is negative.
        # This makes the curve_fit bounds invalid (upper <= lower), raising a
        # ValueError caught by the basic-Gauss handler, leaving psfFitPars
        # None for those rows and exercising the "is None" gaussMoffat branch.
        exp, bbox = makeExposure(self.width, self.height, amp=50.0)
        # Drive a few rows well below the background level.
        exp.image.array[10, :] = -1000.0
        exp.image.array[15, :] = -1000.0
        task = self._makeTask()
        centroid = geom.Point2D(self.width / 2.0, self.height / 2.0)
        task.initialise(exp, centroid, bbox, self.dispersion)
        result = task.getFluxBasic()
        self.assertIs(result, task)
        # The bad rows should have failed the basic Gaussian fit.
        self.assertIsNone(task.psfFitPars[10])
        # Row 0 should still have succeeded.
        self.assertIsNotNone(task.psfFitPars[0])

    def test_getFluxBasic_moffatFitRaises(self):
        # Force the Moffat fitter to raise on every row so the
        # RuntimeError/ValueError handler around it is exercised. (The
        # GaussMoffat handler cannot be driven on every row because later
        # rows reference the previous iteration's fitValsGM, so we leave the
        # GaussMoffat fit working here.)
        exp, bbox = makeExposure(self.width, self.height, amp=50.0)
        task = self._makeTask()
        centroid = geom.Point2D(self.width / 2.0, self.height / 2.0)
        task.initialise(exp, centroid, bbox, self.dispersion)
        with mock.patch.object(task, "moffatFit",
                               side_effect=RuntimeError("boom")):
            result = task.getFluxBasic()
        self.assertIs(result, task)
        # All Moffat fits failed, so that parameter list is unpopulated.
        self.assertTrue(all(p is None for p in task.moffatFitPars))

    def test_getFluxBasic_gaussMoffatRaisesLastRow(self):
        # Make the GaussMoffat fit raise only on the final row so the
        # RuntimeError/ValueError handler around it is exercised without
        # leaving a later row referencing an unbound fitValsGM.
        exp, bbox = makeExposure(self.width, self.height, amp=50.0)
        task = self._makeTask()
        centroid = geom.Point2D(self.width / 2.0, self.height / 2.0)
        task.initialise(exp, centroid, bbox, self.dispersion)

        realFit = task.gaussMoffatFit
        lastRow = self.height - 1
        state = {"calls": 0}

        def sometimesRaise(pixels, footprint, initialPars):
            state["calls"] += 1
            if state["calls"] > lastRow:  # final row only
                raise RuntimeError("boom")
            return realFit(pixels, footprint, initialPars)

        with mock.patch.object(task, "gaussMoffatFit",
                               side_effect=sometimesRaise):
            result = task.getFluxBasic()
        self.assertIs(result, task)
        self.assertIsNone(task.gausMoffatFitPars[lastRow])

    def test_getFluxBasic_debugPlot(self):
        # Exercise the debug.plot block. Note this block references several
        # attributes that the production code never actually sets
        # (self.psf_gauss_flux, self.spectrum.object_name, ...), so we must
        # inject them here for the code path to run at all.
        exp, bbox = makeExposure(self.width, self.height, amp=50.0)
        task = self._makeTask()
        centroid = geom.Point2D(self.width / 2.0, self.height / 2.0)
        task.initialise(exp, centroid, bbox, self.dispersion)
        task.debug = types.SimpleNamespace(display=False, plot="all")
        task.psf_gauss_flux = np.zeros(self.height)
        task.psf_gauss_psfSigma = np.zeros(self.height)
        task.psf_gauss_psfMu = np.zeros(self.height)
        task.spectrum = types.SimpleNamespace(object_name="testStar")
        with mock.patch("lsst.atmospec.extraction.pl.show"):
            result = task.getFluxBasic()
        self.assertIs(result, task)

    def test_initialise_debugDisplayFailure(self):
        # A debug display with an invalid backend must be caught and disable
        # the display without raising (covers the except branch).
        exp, bbox = makeExposure(self.width, self.height)
        task = self._makeTask()
        task.debug = types.SimpleNamespace(
            display=True, displayBackend="not-a-real-backend",
            displayItems=[], plot=False)
        centroid = geom.Point2D(self.width / 2.0, self.height / 2.0)
        task.initialise(exp, centroid, bbox, self.dispersion)
        # The failed display should have been disabled.
        self.assertFalse(task.debug.display)

    def test_initialise_debugDisplaySuccess(self):
        # Patch the real afw.display module so the display path succeeds and
        # the spectrumBgSub display branch is exercised without a real viewer.
        import lsst.afw.display as afwDisp
        fakeDisplayInstance = mock.MagicMock()
        exp, bbox = makeExposure(self.width, self.height)
        task = self._makeTask()
        task.debug = types.SimpleNamespace(
            display=True, displayBackend="fake",
            displayItems=["spectrumBgSub"], plot=False)
        centroid = geom.Point2D(self.width / 2.0, self.height / 2.0)
        with mock.patch.object(afwDisp, "setDefaultBackend"), \
                mock.patch.object(afwDisp, "Display",
                                  return_value=fakeDisplayInstance):
            task.initialise(exp, centroid, bbox, self.dispersion)
        # Display stayed enabled and mtv was invoked (init + spectrumBgSub).
        self.assertTrue(task.debug.display)
        self.assertTrue(fakeDisplayInstance.mtv.called)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
