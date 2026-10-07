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
"""Coverage-oriented test cases for lsst.atmospec.processStar."""

import types
import unittest
from unittest import mock

import numpy as np

import lsst.utils
import lsst.utils.tests
import lsst.afw.image as afwImage
import lsst.geom as geom
import lsst.afw.detection as afwDetect
from lsst.afw.geom.ellipses import Quadrupole
from lsst.pex.config import FieldValidationError

from lsst.atmospec.processStar import (
    ProcessStarTask,
    ProcessStarTaskConfig,
    ProcessStarTaskConnections,
)

MODULE = "lsst.atmospec.processStar"


def makeSourceExposure(width=100, height=100):
    """Build a small ExposureF containing three gaussian sources.

    Returns the exposure. The sources are:
      - a round, faint-ish source at (30, 30)
      - an elongated, bright source at (70, 70)
      - a round, very faint source at (50, 20)
    """
    exp = afwImage.ExposureF(width, height)
    exp.variance.array[:] = 1.0
    arr = exp.image.array
    yy, xx = np.mgrid[0:height, 0:width]

    def addGaussian(x, y, flux, sx, sy):
        g = np.exp(-(((xx - x)**2) / (2 * sx**2)
                     + ((yy - y)**2) / (2 * sy**2)))
        g *= flux / g.sum()
        arr[:] = arr + g

    addGaussian(30, 30, 10000, 2, 2)
    addGaussian(70, 70, 50000, 2, 6)
    addGaussian(50, 20, 500, 2, 2)
    return exp


def fakeFindConfig():
    """A namespace supplying the object-finding config fields.

    These fields are referenced by findObjects/findMainSource but are not
    present on the real config, so tests inject this fake instead.
    """
    return types.SimpleNamespace(
        mainStarNpixMin=5,
        mainStarNsigma=5,
        mainStarGrow=0,
        mainStarGrowIsotropic=True,
        mainSourceFindingMethod='ROUNDEST',
        mainStarFluxCut=1e-15,
        mainStarRoundnessCut=1e9,
    )


class ConfigTestCase(lsst.utils.tests.TestCase):
    """Tests for ProcessStarTaskConfig and connections."""

    def testDefaultsAndSetDefaults(self):
        config = ProcessStarTaskConfig()
        # setDefaults was applied during construction.
        self.assertFalse(config.charImage.doWriteExposure)
        self.assertFalse(config.charImage.doApCorr)
        self.assertFalse(config.charImage.doMeasurePsf)
        self.assertEqual(config.charImage.detection.includeThresholdMultiplier, 3)

    def testValidatePasses(self):
        config = ProcessStarTaskConfig()
        # doFitAtmosphere is False by default so this should not raise.
        config.validate()

    def testValidateFitAtmosphereRaises(self):
        config = ProcessStarTaskConfig()
        config.doFitAtmosphere = True
        with mock.patch(f"{MODULE}.shutil.which", return_value=None):
            with self.assertRaises(FieldValidationError):
                config.validate()

    def testValidateFitAtmosphereOnSpectrogramRaises(self):
        config = ProcessStarTaskConfig()
        config.doFitAtmosphereOnSpectrogram = True
        with mock.patch(f"{MODULE}.shutil.which", return_value=None):
            with self.assertRaises(FieldValidationError):
                config.validate()

    def testValidateWithUvspecPresent(self):
        config = ProcessStarTaskConfig()
        config.doFitAtmosphere = True
        config.doFitAtmosphereOnSpectrogram = True
        with mock.patch(f"{MODULE}.shutil.which", return_value="/usr/bin/uvspec"):
            config.validate()

    def testConnectionsDefault(self):
        config = ProcessStarTaskConfig()
        connections = ProcessStarTaskConnections(config=config)
        self.assertIn("spectrumForwardModelFitParameters", connections.outputs)
        self.assertNotIn("spectrumLibradtranFitParameters", connections.outputs)
        self.assertNotIn("spectrogramLibradtranFitParameters", connections.outputs)

    def testConnectionsToggles(self):
        config = ProcessStarTaskConfig()
        config.doFullForwardModelDeconvolution = False
        connections = ProcessStarTaskConnections(config=config)
        self.assertNotIn("spectrumForwardModelFitParameters", connections.outputs)


class ProcessStarTaskTestCase(lsst.utils.tests.TestCase):
    """Tests for ProcessStarTask methods."""

    def setUp(self):
        self.config = ProcessStarTaskConfig()
        self.task = ProcessStarTask(config=self.config)

    def testInit(self):
        self.assertTrue(hasattr(self.task, "isr"))
        self.assertTrue(hasattr(self.task, "charImage"))

    def testGetEllipticity(self):
        # A round shape has (near) zero ellipticity.
        roundShape = Quadrupole(4.0, 4.0, 0.0)
        self.assertAlmostEqual(self.task._getEllipticity(roundShape), 0.0)
        # An elongated shape has a positive ellipticity.
        elongated = Quadrupole(10.0, 2.0, 0.0)
        self.assertGreater(self.task._getEllipticity(elongated), 0.0)

    def testFindObjects(self):
        exp = makeSourceExposure()
        self.task.config = fakeFindConfig()
        fpSet = self.task.findObjects(exp)
        self.assertIsInstance(fpSet, afwDetect.FootprintSet)
        self.assertGreaterEqual(len(fpSet.getFootprints()), 1)

    def testFindObjectsWithSigmaAndGrow(self):
        exp = makeSourceExposure()
        cfg = fakeFindConfig()
        cfg.mainStarGrow = 2
        self.task.config = cfg
        # Pass explicit nSigma; grow taken from config (>0 branch).
        fpSet = self.task.findObjects(exp, nSigma=5)
        self.assertIsInstance(fpSet, afwDetect.FootprintSet)

    def testGetRoundestObject(self):
        exp = makeSourceExposure()
        self.task.config = fakeFindConfig()
        fpSet = self.task.findObjects(exp)
        source = self.task.getRoundestObject(fpSet, exp, fluxCut=1e-15)
        self.assertIsInstance(source, afwDetect.Footprint)

    def testGetBrightestObject(self):
        exp = makeSourceExposure()
        self.task.config = fakeFindConfig()
        fpSet = self.task.findObjects(exp)
        source = self.task.getBrightestObject(fpSet, exp, roundnessCut=1e9)
        self.assertIsInstance(source, afwDetect.Footprint)

    def testFindMainSourceRoundest(self):
        exp = makeSourceExposure()
        cfg = fakeFindConfig()
        cfg.mainSourceFindingMethod = 'ROUNDEST'
        self.task.config = cfg
        centroid = self.task.findMainSource(exp)
        self.assertEqual(len(centroid), 2)

    def testFindMainSourceBrightest(self):
        exp = makeSourceExposure()
        cfg = fakeFindConfig()
        cfg.mainSourceFindingMethod = 'BRIGHTEST'
        self.task.config = cfg
        centroid = self.task.findMainSource(exp)
        self.assertEqual(len(centroid), 2)

    def testFindMainSourceInvalid(self):
        exp = makeSourceExposure()
        cfg = fakeFindConfig()
        cfg.mainSourceFindingMethod = 'NONSENSE'
        self.task.config = cfg
        with self.assertRaises(RuntimeError):
            self.task.findMainSource(exp)

    def testUpdateMetadataWithCentroid(self):
        exp = afwImage.ExposureF(10, 10)
        vi = afwImage.VisitInfo(boresightAirmass=1.5)
        exp.getInfo().setVisitInfo(vi)
        self.task.updateMetadata(exp, centroid=(12.0, 34.0))
        md = exp.getMetadata()
        self.assertEqual(md['OBJECTX'], 12.0)
        self.assertEqual(md['OBJECTY'], 34.0)
        self.assertEqual(md['AIRMASS'], 1.5)
        self.assertIn('HA', md)

    def testUpdateMetadataWithoutCentroid(self):
        exp = afwImage.ExposureF(10, 10)
        vi = afwImage.VisitInfo(boresightAirmass=1.2)
        exp.getInfo().setVisitInfo(vi)
        self.task.updateMetadata(exp)
        md = exp.getMetadata()
        self.assertIsNone(md['OBJECTX'])
        self.assertIsNone(md['OBJECTY'])

    def testGetNormalizedTargetNameMapped(self):
        # 'ETA1DOR' is mapped in nameMappings.txt.
        result = self.task.getNormalizedTargetName('spec:ETA1DOR')
        self.assertEqual(result, 'HD42525')

    def testGetNormalizedTargetNameUnmapped(self):
        result = self.task.getNormalizedTargetName('HD999999')
        self.assertEqual(result, 'HD999999')

    def testGetSpectractorTargetSettingAutoExact(self):
        self.task.config = types.SimpleNamespace(targetCentroidMethod='auto')
        result = self.task._getSpectractorTargetSetting({'astrometricMatch': True})
        self.assertEqual(result, 'guess')

    def testGetSpectractorTargetSettingAutoFit(self):
        self.task.config = types.SimpleNamespace(targetCentroidMethod='auto')
        result = self.task._getSpectractorTargetSetting({'astrometricMatch': False})
        self.assertEqual(result, 'fit')

    def testGetSpectractorTargetSettingExact(self):
        self.task.config = types.SimpleNamespace(targetCentroidMethod='exact')
        self.assertEqual(self.task._getSpectractorTargetSetting({}), 'guess')

    def testGetSpectractorTargetSettingFallthrough(self):
        self.task.config = types.SimpleNamespace(targetCentroidMethod='WCS')
        self.assertEqual(self.task._getSpectractorTargetSetting({}), 'WCS')

    def testLoadStarNames(self):
        names = self.task.loadStarNames()
        self.assertIsInstance(names, list)
        self.assertTrue(len(names) > 0)

    def testFlatfield(self):
        exp = afwImage.ExposureF(5, 5)
        self.assertIs(self.task.flatfield(exp, None), exp)

    def testRepairCosmics(self):
        exp = afwImage.ExposureF(5, 5)
        self.assertIs(self.task.repairCosmics(exp, None), exp)

    def testMeasureSpectrum(self):
        self.task.extraction = mock.MagicMock()
        sentinel = object()
        self.task.extraction.getFluxBasic.return_value = sentinel
        result = self.task.measureSpectrum("exp", "cen", "bbox", "disp")
        self.assertIs(result, sentinel)
        self.task.extraction.initialise.assert_called_once()

    def testCalcSpectrumBBoxPositiveOrder(self):
        exp = afwImage.ExposureF(1000, 6000)
        bbox = self.task.calcSpectrumBBox(exp, (500, 500), 40, order='+1')
        self.assertIsInstance(bbox, geom.Box2I)
        self.assertEqual(bbox.getMinY(), 600)

    def testCalcSpectrumBBoxNegativeOrder(self):
        exp = afwImage.ExposureF(1000, 6000)
        bbox = self.task.calcSpectrumBBox(exp, (500, 5500), 40, order='-1')
        self.assertIsInstance(bbox, geom.Box2I)
        # yStart is clamped to a value >= 0.
        self.assertGreaterEqual(bbox.getMinY(), 0)

    def testPauseDefault(self):
        self.task.debug = types.SimpleNamespace(pauseOnDisplay=False)
        self.assertIsNone(self.task.pause())

    def testPauseWithInput(self):
        self.task.debug = types.SimpleNamespace(pauseOnDisplay=True)
        with mock.patch("builtins.input", return_value=""):
            self.assertIsNone(self.task.pause())

    def testRunQuantum(self):
        butlerQC = mock.MagicMock()
        inputRefs = mock.MagicMock()
        outputRefs = mock.MagicMock()
        butlerQC.get.return_value = {
            'inputExp': 'exp',
            'inputCentroid': 'cen',
        }
        self.task.run = mock.MagicMock(return_value="outputs")
        self.task.runQuantum(butlerQC, inputRefs, outputRefs)
        self.task.run.assert_called_once()
        _, kwargs = self.task.run.call_args
        self.assertIn('dataIdDict', kwargs)
        butlerQC.put.assert_called_once_with("outputs", outputRefs)

    def _makeRunExposure(self, objectName='HD12345'):
        exp = afwImage.ExposureF(10, 10)
        vi = afwImage.VisitInfo(boresightAirmass=1.5, object=objectName)
        exp.getInfo().setVisitInfo(vi)
        return exp

    def _runWithMocks(self, exp, inputCentroid, grating='other'):
        with mock.patch(f"{MODULE}.isDispersedExp", return_value=True), \
                mock.patch(f"{MODULE}.getLinearStagePosition", return_value=10.0), \
                mock.patch(f"{MODULE}.getFilterAndDisperserFromExp",
                           return_value=('SDSSr', grating)), \
                mock.patch(f"{MODULE}.SpectractorShim") as shim:
            instance = shim.return_value
            instance.run.return_value = mock.MagicMock()
            result = self.task.run(inputExp=exp, inputCentroid=inputCentroid,
                                   dataIdDict={'visit': 1})
            return result, shim, instance

    def testRunNotDispersedRaises(self):
        exp = self._makeRunExposure()
        with mock.patch(f"{MODULE}.isDispersedExp", return_value=False):
            with self.assertRaises(RuntimeError):
                self.task.run(inputExp=exp, inputCentroid={}, dataIdDict={})

    def testRunBadObjectRaises(self):
        exp = self._makeRunExposure(objectName='Test')
        with mock.patch(f"{MODULE}.isDispersedExp", return_value=True), \
                mock.patch(f"{MODULE}.getLinearStagePosition", return_value=10.0), \
                mock.patch(f"{MODULE}.getFilterAndDisperserFromExp",
                           return_value=('SDSSr', 'other')):
            with self.assertRaises(ValueError):
                self.task.run(inputExp=exp,
                              inputCentroid={'astrometricMatch': True,
                                             'centroid': (1, 2)},
                              dataIdDict={})

    def testRunWithDictCentroid(self):
        exp = self._makeRunExposure()
        inputCentroid = {'astrometricMatch': True, 'centroid': (5.0, 6.0)}
        result, shim, instance = self._runWithMocks(exp, inputCentroid)
        shim.assert_called_once()
        instance.run.assert_called_once()
        self.assertTrue(hasattr(result, "spectractorSpectrum"))

    def testRunWithTupleCentroid(self):
        # A raw tuple centroid is only meaningful when the centroid method is
        # not 'auto' (auto expects the dict form), so use an 'exact' config.
        config = ProcessStarTaskConfig()
        config.targetCentroidMethod = 'exact'
        self.task = ProcessStarTask(config=config)
        exp = self._makeRunExposure()
        inputCentroid = (5.0, 6.0)
        result, shim, instance = self._runWithMocks(exp, inputCentroid)
        instance.run.assert_called_once()

    def testRunHologramGrating(self):
        exp = self._makeRunExposure()
        inputCentroid = {'astrometricMatch': False, 'centroid': (5.0, 6.0)}
        result, shim, instance = self._runWithMocks(exp, inputCentroid,
                                                    grating='holo4_003')
        # 4mm window added to the linear stage position of 10.0
        _, kwargs = shim.call_args
        self.assertEqual(kwargs['paramOverrides']['DISTANCE2CCD'], 14.0)

    def testRunForceObjectName(self):
        config = ProcessStarTaskConfig()
        config.forceObjectName = 'HD54321'
        task = ProcessStarTask(config=config)
        exp = self._makeRunExposure()
        inputCentroid = {'astrometricMatch': True, 'centroid': (5.0, 6.0)}
        with mock.patch(f"{MODULE}.isDispersedExp", return_value=True), \
                mock.patch(f"{MODULE}.getLinearStagePosition", return_value=10.0), \
                mock.patch(f"{MODULE}.getFilterAndDisperserFromExp",
                           return_value=('SDSSr', 'other')), \
                mock.patch(f"{MODULE}.SpectractorShim") as shim:
            shim.return_value.run.return_value = mock.MagicMock()
            task.run(inputExp=exp, inputCentroid=inputCentroid,
                     dataIdDict={'visit': 1})
            args, _ = shim.return_value.run.call_args
            # target is the final positional argument passed to shim.run
            self.assertEqual(args[3], 'HD54321')

    def testRunAstrometrySuccess(self):
        exp = afwImage.ExposureF(10, 10)
        with mock.patch(f"{MODULE}.ReferenceObjectLoader"), \
                mock.patch(f"{MODULE}.FitAffineWcsTask"), \
                mock.patch(f"{MODULE}.AstrometryTask") as astromTask:
            solver = astromTask.return_value
            astromResult = mock.MagicMock()
            astromResult.scatterOnSky.asArcseconds.return_value = 0.5
            solver.run.return_value = astromResult
            result = self.task.runAstrometry(butler=None, exp=exp, icSrc=None)
            self.assertIs(result, astromResult)

    def testRunAstrometryPoorScatter(self):
        exp = afwImage.ExposureF(10, 10)
        with mock.patch(f"{MODULE}.ReferenceObjectLoader"), \
                mock.patch(f"{MODULE}.FitAffineWcsTask"), \
                mock.patch(f"{MODULE}.AstrometryTask") as astromTask:
            solver = astromTask.return_value
            astromResult = mock.MagicMock()
            astromResult.scatterOnSky.asArcseconds.return_value = 5.0
            solver.run.return_value = astromResult
            result = self.task.runAstrometry(butler=None, exp=exp, icSrc=None)
            self.assertIsNone(result)

    def testRunAstrometryFailure(self):
        exp = afwImage.ExposureF(10, 10)
        with mock.patch(f"{MODULE}.ReferenceObjectLoader"), \
                mock.patch(f"{MODULE}.FitAffineWcsTask"), \
                mock.patch(f"{MODULE}.AstrometryTask") as astromTask:
            solver = astromTask.return_value
            solver.run.side_effect = RuntimeError("boom")
            result = self.task.runAstrometry(butler=None, exp=exp, icSrc=None)
            self.assertIsNone(result)


class InitDebugTestCase(lsst.utils.tests.TestCase):
    """Exercise the debug-display branches of ProcessStarTask.__init__."""

    def _makeDebugInfo(self, **kwargs):
        defaults = dict(enabled=False, display=False, displayBackend='virtualDevice',
                        notHeadless=False, pauseOnDisplay=False)
        defaults.update(kwargs)
        return types.SimpleNamespace(**defaults)

    def testInitDebugEnabledNoDisplay(self):
        debug = self._makeDebugInfo(enabled=True, display=False, notHeadless=True)
        with mock.patch(f"{MODULE}.lsstDebug.Info", return_value=debug):
            task = ProcessStarTask(config=ProcessStarTaskConfig())
        self.assertTrue(task.debug.enabled)

    def testInitDebugDisplaySuccess(self):
        import lsst.afw.display as afwDisp
        debug = self._makeDebugInfo(enabled=True, display=True, notHeadless=False)
        with mock.patch(f"{MODULE}.lsstDebug.Info", return_value=debug), \
                mock.patch.object(afwDisp, "setDefaultBackend"), \
                mock.patch.object(afwDisp, "setDefaultMaskTransparency"), \
                mock.patch.object(afwDisp, "Display") as displayMock:
            task = ProcessStarTask(config=ProcessStarTaskConfig())
        self.assertTrue(task.debug.display)
        displayMock.assert_called()

    def testInitDebugDisplayNameError(self):
        # Force the display setup to raise NameError, hitting the except block.
        import lsst.afw.display as afwDisp
        debug = self._makeDebugInfo(enabled=True, display=True, notHeadless=False)
        with mock.patch(f"{MODULE}.lsstDebug.Info", return_value=debug), \
                mock.patch.object(afwDisp, "setDefaultBackend",
                                  side_effect=NameError("boom")):
            task = ProcessStarTask(config=ProcessStarTaskConfig())
        # display is disabled after the NameError.
        self.assertFalse(task.debug.display)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
