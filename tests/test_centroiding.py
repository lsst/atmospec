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
"""Test cases for the atmospec SingleStarCentroidTask."""

import unittest
from unittest import mock

import lsst.utils
import lsst.utils.tests
import lsst.geom as geom
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
from lsst.pipe.base.task import TaskError
from lsst.pipe.tasks.quickFrameMeasurement import QuickFrameMeasurementTask
from lsst.pipe.tasks.peekExposure import PeekExposureTask

from lsst.atmospec.centroiding import SingleStarCentroidTask, SingleStarCentroidTaskConfig


def makeExposureWithWcs(target="HD 12345"):
    """Build a small ExposureF with a valid WCS and a visitInfo object."""
    exp = afwImage.ExposureF(16, 16)
    wcs = afwGeom.makeSkyWcs(geom.Point2D(0, 0),
                             geom.SpherePoint(10*geom.degrees, 20*geom.degrees),
                             afwGeom.makeCdMatrix(1e-4*geom.degrees))
    exp.setWcs(wcs)
    exp.info.setVisitInfo(afwImage.VisitInfo(object=target))
    return exp


class SingleStarCentroidConfigTestCase(lsst.utils.tests.TestCase):
    """Tests for the task config, including setDefaults and validate."""

    def testSetDefaultsAndValidate(self):
        # The default fallback task is PeekExposureTask, which is valid.
        config = SingleStarCentroidTaskConfig()
        self.assertEqual(config.astromRefObjLoader.pixelMargin, 1000)
        self.assertEqual(config.referenceFilterOverride, "phot_g_mean")
        # Should not raise.
        config.validate()

    def testValidateWithQuickFrameMeasurement(self):
        config = SingleStarCentroidTaskConfig()
        config.centroidingFallbackTask.retarget(QuickFrameMeasurementTask)
        # quickFrameMeasurementTask is an accepted fallback.
        config.validate()

    def testValidateUnknownFallbackRaises(self):
        config = SingleStarCentroidTaskConfig()
        # Retarget to a task whose _DefaultName is not accepted.
        config.centroidingFallbackTask.retarget(SingleStarCentroidTask)
        with self.assertRaises(ValueError):
            config.validate()


class SingleStarCentroidTaskTestCase(lsst.utils.tests.TestCase):
    """Tests for the SingleStarCentroidTask itself."""

    def makeTask(self):
        config = SingleStarCentroidTaskConfig()
        return SingleStarCentroidTask(config=config)

    def testInit(self):
        task = self.makeTask()
        self.assertIsInstance(task, SingleStarCentroidTask)
        self.assertIsInstance(task.centroidingFallbackTask, PeekExposureTask)

    def testRunSuccessfulFit(self):
        task = self.makeTask()
        exp = makeExposureWithWcs()

        astromResult = mock.MagicMock()
        astromResult.scatterOnSky.asArcseconds.return_value = 0.5
        task.astrometry = mock.MagicMock()
        task.astrometry.run.return_value = astromResult

        with mock.patch("lsst.atmospec.centroiding.getTargetCentroidFromWcs",
                        return_value=geom.Point2D(3.0, 4.0)):
            result = task.run(inputExp=exp, inputSources=mock.MagicMock())

        self.assertTrue(result.atmospecCentroid["astrometricMatch"])
        self.assertEqual(result.atmospecCentroid["centroid"], (3.0, 4.0))

    def testRunSuccessfulFitNoCentroidFallsBack(self):
        task = self.makeTask()
        exp = makeExposureWithWcs()

        astromResult = mock.MagicMock()
        astromResult.scatterOnSky.asArcseconds.return_value = 0.5
        task.astrometry = mock.MagicMock()
        task.astrometry.run.return_value = astromResult

        # WCS lookup returns None, forcing the fallback path.
        peek = mock.MagicMock(spec=PeekExposureTask)
        peek.run.return_value = pipeBase.Struct(brightestCentroid=geom.Point2D(7.0, 8.0))
        task.centroidingFallbackTask = peek

        with mock.patch("lsst.atmospec.centroiding.getTargetCentroidFromWcs", return_value=None):
            result = task.run(inputExp=exp, inputSources=mock.MagicMock())

        self.assertFalse(result.atmospecCentroid["astrometricMatch"])
        self.assertEqual(result.atmospecCentroid["centroid"], (7.0, 8.0))

    def testRunScatterTooLargeFallsBack(self):
        task = self.makeTask()
        exp = makeExposureWithWcs()

        astromResult = mock.MagicMock()
        astromResult.scatterOnSky.asArcseconds.return_value = 5.0  # >= 1, so not successful
        task.astrometry = mock.MagicMock()
        task.astrometry.run.return_value = astromResult

        peek = mock.MagicMock(spec=PeekExposureTask)
        peek.run.return_value = pipeBase.Struct(brightestCentroid=geom.Point2D(1.0, 2.0))
        task.centroidingFallbackTask = peek

        result = task.run(inputExp=exp, inputSources=mock.MagicMock())
        self.assertFalse(result.atmospecCentroid["astrometricMatch"])
        self.assertEqual(result.atmospecCentroid["centroid"], (1.0, 2.0))

    def testRunAstrometryRaisesFallsBack(self):
        task = self.makeTask()
        exp = makeExposureWithWcs()
        originalWcs = exp.getWcs()

        task.astrometry = mock.MagicMock()
        task.astrometry.run.side_effect = TaskError("no solution")

        peek = mock.MagicMock(spec=PeekExposureTask)
        peek.run.return_value = pipeBase.Struct(brightestCentroid=geom.Point2D(9.0, 10.0))
        task.centroidingFallbackTask = peek

        result = task.run(inputExp=exp, inputSources=mock.MagicMock())
        self.assertFalse(result.atmospecCentroid["astrometricMatch"])
        self.assertEqual(result.atmospecCentroid["centroid"], (9.0, 10.0))
        # The WCS should have been restored after the failure.
        self.assertIsNotNone(exp.getWcs())
        self.assertEqual(exp.getWcs(), originalWcs)

    def testRunFallbackTaskQuickFrameMeasurement(self):
        task = self.makeTask()
        qfm = mock.MagicMock(spec=QuickFrameMeasurementTask)
        qfm.run.return_value = pipeBase.Struct(brightestObjCentroid=(11.0, 12.0))
        task.centroidingFallbackTask = qfm

        centroid = task.runFallbackTask(makeExposureWithWcs())
        self.assertEqual(centroid, (11.0, 12.0))

    def testRunFallbackTaskPeekExposure(self):
        task = self.makeTask()
        peek = mock.MagicMock(spec=PeekExposureTask)
        peek.run.return_value = pipeBase.Struct(brightestCentroid=geom.Point2D(13.0, 14.0))
        task.centroidingFallbackTask = peek

        centroid = task.runFallbackTask(makeExposureWithWcs())
        self.assertEqual(centroid, (13.0, 14.0))

    def testRunFallbackTaskUnsupportedRaises(self):
        task = self.makeTask()
        # A mock that is neither QFM nor PeekExposureTask.
        task.centroidingFallbackTask = mock.MagicMock()
        with self.assertRaises(ValueError):
            task.runFallbackTask(makeExposureWithWcs())

    def testRunQuantum(self):
        task = self.makeTask()
        task.astrometry = mock.MagicMock()

        exp = makeExposureWithWcs()
        sources = mock.MagicMock()

        butlerQC = mock.MagicMock()
        butlerQC.get.return_value = {"inputExp": exp,
                                     "inputSources": sources,
                                     "astromRefCat": [mock.MagicMock()]}

        inputRefs = mock.MagicMock()
        inputRefs.astromRefCat = [mock.MagicMock()]
        outputRefs = mock.MagicMock()

        fakeOutputs = pipeBase.Struct(atmospecCentroid={"centroid": (1.0, 2.0),
                                                        "astrometricMatch": True})

        with mock.patch("lsst.atmospec.centroiding.ReferenceObjectLoader") as mockLoader, \
                mock.patch.object(task, "run", return_value=fakeOutputs) as mockRun:
            task.runQuantum(butlerQC, inputRefs, outputRefs)

        mockLoader.assert_called_once()
        task.astrometry.setRefObjLoader.assert_called_once()
        mockRun.assert_called_once_with(inputExp=exp, inputSources=sources)
        butlerQC.put.assert_called_once_with(fakeOutputs, outputRefs)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
