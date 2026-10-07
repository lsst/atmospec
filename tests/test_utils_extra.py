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
"""Additional test cases for atmospec.utils, targeting full coverage."""

import unittest
from unittest import mock

import numpy as np
import astropy.units as u
from astropy.table import Table

import lsst.utils
import lsst.utils.tests
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.cameraGeom as cameraGeom
import lsst.geom as geom
import lsst.daf.base as dafBase
import lsst.daf.butler as dafButler
import lsst.pex.config as pexConfig

from lsst.obs.lsst.translators.lsst import FILTER_DELIMITER
from lsst.atmospec.utils import (
    makeGainFlat,
    isExposureTrimmed,
    getAmpReadNoiseFromRawExp,
    gainFromFlatPair,
    rotateExposure,
    getLinearStagePosition,
    getFilterAndDisperserFromExp,
    isDispersedExp,
    isDispersedDataId,
    simbadLocationForTarget,
    vizierLocationForTarget,
    getTargetCentroidFromWcs,
    runNotebook,
)


def makeTestDetector(numAmps=2, ampWidth=10, ampHeight=10, prescan=3, oscan=5):
    """Build a small camera detector with real trimmed and raw bounding boxes.

    Each amp has a raw layout of prescan + ampWidth + oscan columns.
    """
    camBuilder = cameraGeom.Camera.Builder('testCam')
    detBuilder = camBuilder.add('det0', 0)
    detBuilder.setSerial('det0-serial')
    detBuilder.setBBox(geom.Box2I(geom.Point2I(0, 0),
                                  geom.Extent2I(numAmps*ampWidth, ampHeight)))
    rawWidth = prescan + ampWidth + oscan
    for i in range(numAmps):
        ampBuilder = cameraGeom.Amplifier.Builder()
        ampBuilder.setName(f'amp{i}')
        ampBuilder.setBBox(geom.Box2I(geom.Point2I(i*ampWidth, 0),
                                      geom.Extent2I(ampWidth, ampHeight)))
        ampBuilder.setRawBBox(geom.Box2I(geom.Point2I(0, 0),
                                         geom.Extent2I(rawWidth, ampHeight)))
        ampBuilder.setRawDataBBox(geom.Box2I(geom.Point2I(prescan, 0),
                                             geom.Extent2I(ampWidth, ampHeight)))
        ampBuilder.setRawHorizontalOverscanBBox(
            geom.Box2I(geom.Point2I(prescan + ampWidth, 0),
                       geom.Extent2I(oscan, ampHeight)))
        ampBuilder.setRawXYOffset(geom.Extent2I(i*rawWidth, 0))
        ampBuilder.setGain(1.0)
        ampBuilder.setReadNoise(1.0)
        detBuilder.append(ampBuilder)
    cam = camBuilder.finish()
    return cam['det0']


def makeTrimmedExp(det, fillValue=0.0):
    """Make a trimmed exposure with the given detector."""
    exp = afwImage.ExposureF(det.getBBox())
    exp.setDetector(det)
    if fillValue:
        exp.image.array[:, :] = fillValue
    return exp


def makeRawExp(det):
    """Make a raw (untrimmed) exposure with the given detector."""
    rawWidth = sum(amp.getRawBBox().getWidth() for amp in det)
    rawHeight = det[0].getRawBBox().getHeight()
    exp = afwImage.ExposureF(geom.Box2I(geom.Point2I(0, 0),
                                        geom.Extent2I(rawWidth, rawHeight)))
    exp.setDetector(det)
    return exp


def makeWcs(crpix=(15, 15), crval=(30.0, 10.0)):
    crpixPoint = geom.Point2D(*crpix)
    crvalPoint = geom.SpherePoint(crval[0]*geom.degrees, crval[1]*geom.degrees)
    cdMatrix = afwGeom.makeCdMatrix(scale=0.2*geom.arcseconds)
    return afwGeom.makeSkyWcs(crpixPoint, crvalPoint, cdMatrix)


def makeHipTable():
    """Build a table mimicking the Vizier Hipparcos2 ('I/311/hip2') result."""
    star = Table()
    star['RArad'] = [30.0]*u.deg
    star['DErad'] = [10.0]*u.deg
    star['pmRA'] = [1.0]*(u.mas/u.yr)
    star['pmDE'] = [1.0]*(u.mas/u.yr)
    star['Plx'] = [10.0]*u.mas
    return star


class MakeGainFlatTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.det = makeTestDetector()
        self.exp = makeTrimmedExp(self.det)
        self.gainDict = {amp.getName(): float(i + 2)
                         for i, amp in enumerate(self.det)}

    def test_makeGainFlat_normal(self):
        flat = makeGainFlat(self.exp, self.gainDict, invertGains=False)
        for amp in self.det:
            vals = flat[amp.getBBox()].image.array
            self.assertTrue(np.all(vals == self.gainDict[amp.getName()]))
        self.assertTrue(np.all(flat.maskedImage.mask.array == 0))
        self.assertTrue(np.all(flat.maskedImage.variance.array == 0.0))

    def test_makeGainFlat_inverted(self):
        flat = makeGainFlat(self.exp, self.gainDict, invertGains=True)
        for amp in self.det:
            vals = flat[amp.getBBox()].image.array
            expected = 1.0/self.gainDict[amp.getName()]
            self.assertTrue(np.allclose(vals, expected))

    def test_makeGainFlat_mismatchedKeys(self):
        badDict = {'notAnAmp': 1.0}
        with self.assertRaises(AssertionError):
            makeGainFlat(self.exp, badDict)


class IsExposureTrimmedTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.det = makeTestDetector()

    def test_trimmed(self):
        exp = makeTrimmedExp(self.det)
        self.assertTrue(isExposureTrimmed(exp))

    def test_untrimmed(self):
        exp = makeRawExp(self.det)
        self.assertFalse(isExposureTrimmed(exp))


class GetAmpReadNoiseTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.det = makeTestDetector()

    def test_raisesOnTrimmed(self):
        exp = makeTrimmedExp(self.det)
        with self.assertRaises(RuntimeError):
            getAmpReadNoiseFromRawExp(exp, 0)

    def test_noise_noBorder(self):
        raw = makeRawExp(self.det)
        rng = np.random.RandomState(1234)
        raw.image.array[:, :] = rng.normal(0, 5, raw.image.array.shape)
        noise = getAmpReadNoiseFromRawExp(raw, 0, nOscanBorderPix=0)
        self.assertGreater(noise, 0.0)

    def test_noise_withBorder(self):
        # need a large enough overscan so that cropping the border leaves data
        det = makeTestDetector(ampWidth=10, ampHeight=20, oscan=12)
        raw = makeRawExp(det)
        rng = np.random.RandomState(1234)
        raw.image.array[:, :] = rng.normal(0, 5, raw.image.array.shape)
        noise = getAmpReadNoiseFromRawExp(raw, 0, nOscanBorderPix=2)
        self.assertGreater(noise, 0.0)


class GainFromFlatPairTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.det = makeTestDetector()
        self.flat1 = makeTrimmedExp(self.det)
        self.flat2 = makeTrimmedExp(self.det)
        rng = np.random.RandomState(42)
        self.flat1.image.array[:, :] = 1000.0 + rng.normal(0, 5, self.flat1.image.array.shape)
        self.flat2.image.array[:, :] = 1000.0 + rng.normal(0, 5, self.flat2.image.array.shape)
        self.raw = makeRawExp(self.det)
        self.raw.image.array[:, :] = rng.normal(0, 5, self.raw.image.array.shape)

    def test_noCorrection(self):
        gains = gainFromFlatPair(self.flat1, self.flat2, correctionType=None)
        self.assertEqual(set(gains.keys()),
                         set(a.getName() for a in self.det))
        for v in gains.values():
            self.assertGreater(v, 0.0)

    def test_simpleCorrection(self):
        gains = gainFromFlatPair(self.flat1, self.flat2, correctionType='simple',
                                 rawExpForNoiseCalc=self.raw)
        self.assertEqual(len(gains), len(list(self.det)))

    def test_fullCorrection(self):
        gains = gainFromFlatPair(self.flat1, self.flat2, correctionType='full',
                                 rawExpForNoiseCalc=self.raw)
        self.assertEqual(len(gains), len(list(self.det)))

    def test_unknownCorrectionType(self):
        with self.assertRaises(RuntimeError):
            gainFromFlatPair(self.flat1, self.flat2, correctionType='bogus')

    def test_correctionWithoutRawExp(self):
        with self.assertRaises(RuntimeError):
            gainFromFlatPair(self.flat1, self.flat2, correctionType='simple')


class RotateExposureTestCase(lsst.utils.tests.TestCase):
    def test_noWcs(self):
        exp = afwImage.ExposureF(10, 10)
        rotated = rotateExposure(exp, 30)
        self.assertEqual(rotated.getDimensions(), exp.getDimensions())
        # confirm it is a copy, not the same object
        self.assertIsNot(rotated, exp)

    def test_noWcs_customLogger(self):
        import logging
        exp = afwImage.ExposureF(10, 10)
        logger = logging.getLogger('test.rotate')
        rotated = rotateExposure(exp, 30, logger=logger)
        self.assertEqual(rotated.getDimensions(), exp.getDimensions())

    def test_rotateExposureF(self):
        exp = afwImage.ExposureF(30, 30)
        exp.setWcs(makeWcs())
        rotated = rotateExposure(exp, 90)
        self.assertIsNotNone(rotated.getWcs())

    def test_rotateExposureU(self):
        exp = afwImage.ExposureU(30, 30)
        exp.setWcs(makeWcs())
        rotated = rotateExposure(exp, 90)
        # gets converted to ExposureF internally
        self.assertIsInstance(rotated, afwImage.ExposureF)


class GetLinearStagePositionTestCase(lsst.utils.tests.TestCase):
    def test_withPosition(self):
        exp = afwImage.ExposureF(10, 10)
        md = dafBase.PropertyList()
        md['LINSPOS'] = 5.0
        exp.setMetadata(md)
        self.assertEqual(getLinearStagePosition(exp), 120.0)

    def test_withNonePosition(self):
        exp = afwImage.ExposureF(10, 10)
        md = dafBase.PropertyList()
        md.set('LINSPOS', None)
        exp.setMetadata(md)
        self.assertEqual(getLinearStagePosition(exp), 115)

    def test_withoutPosition(self):
        exp = afwImage.ExposureF(10, 10)
        exp.setMetadata(dafBase.PropertyList())
        self.assertEqual(getLinearStagePosition(exp), 115)


class GetFilterAndDisperserTestCase(lsst.utils.tests.TestCase):
    def test_withDelimiter(self):
        exp = afwImage.ExposureF(10, 10)
        exp.setFilter(afwImage.FilterLabel(
            band='r', physical='SDSSr' + FILTER_DELIMITER + 'ronchi170lpmm'))
        filt, grating = getFilterAndDisperserFromExp(exp)
        self.assertEqual(filt, 'SDSSr')
        self.assertEqual(grating, 'ronchi170lpmm')

    def test_withoutDelimiter(self):
        exp = afwImage.ExposureF(10, 10)
        exp.setFilter(afwImage.FilterLabel(band='r', physical='SDSSr'))
        md = dafBase.PropertyList()
        md['GRATING'] = 'ronchi170lpmm'
        exp.setMetadata(md)
        filt, grating = getFilterAndDisperserFromExp(exp)
        self.assertEqual(filt, 'SDSSr')
        self.assertEqual(grating, 'ronchi170lpmm')


class IsDispersedExpTestCase(lsst.utils.tests.TestCase):
    def _makeExp(self, physical):
        exp = afwImage.ExposureF(10, 10)
        exp.setFilter(afwImage.FilterLabel(band='r', physical=physical))
        return exp

    def test_dispersed(self):
        exp = self._makeExp('SDSSr' + FILTER_DELIMITER + 'ronchi170lpmm')
        self.assertTrue(isDispersedExp(exp))

    def test_notDispersed(self):
        exp = self._makeExp('SDSSr' + FILTER_DELIMITER + 'empty')
        self.assertFalse(isDispersedExp(exp))

    def test_noDelimiter(self):
        exp = self._makeExp('SDSSr')
        with self.assertRaises(RuntimeError):
            isDispersedExp(exp)


class IsDispersedDataIdTestCase(lsst.utils.tests.TestCase):
    def _makeButler(self, physicalFilter, nRecords=1):
        butler = mock.MagicMock(spec=dafButler.Butler)
        records = []
        for _ in range(nRecords):
            rec = mock.MagicMock()
            rec.physical_filter = physicalFilter
            records.append(rec)
        butler.registry.queryDimensionRecords.return_value = records
        return butler

    def test_notAButler(self):
        with self.assertRaises(RuntimeError):
            isDispersedDataId({'day_obs': 1, 'seq_num': 1}, object())

    def test_dispersed(self):
        butler = self._makeButler('SDSSr' + FILTER_DELIMITER + 'ronchi170lpmm')
        self.assertTrue(isDispersedDataId({'day_obs': 20200101, 'seq_num': 5}, butler))

    def test_empty(self):
        butler = self._makeButler('SDSSr' + FILTER_DELIMITER + 'empty')
        self.assertFalse(isDispersedDataId({'day_obs': 20200101, 'seq_num': 5}, butler))

    def test_exposureDottedKeys(self):
        butler = self._makeButler('SDSSr' + FILTER_DELIMITER + 'ronchi170lpmm')
        dataId = {'exposure.day_obs': 20200101, 'exposure.seq_num': 5}
        self.assertTrue(isDispersedDataId(dataId, butler))

    def test_missingDayObs(self):
        butler = self._makeButler('SDSSr' + FILTER_DELIMITER + 'empty')
        with self.assertRaises(AssertionError):
            isDispersedDataId({'seq_num': 5}, butler)

    def test_missingSeqNum(self):
        butler = self._makeButler('SDSSr' + FILTER_DELIMITER + 'empty')
        with self.assertRaises(AssertionError):
            isDispersedDataId({'day_obs': 20200101}, butler)

    def test_multipleRecords(self):
        butler = self._makeButler('SDSSr' + FILTER_DELIMITER + 'empty', nRecords=2)
        with self.assertRaises(AssertionError):
            isDispersedDataId({'day_obs': 20200101, 'seq_num': 5}, butler)

    def test_noDelimiter(self):
        butler = self._makeButler('SDSSr')
        with self.assertRaises(RuntimeError):
            isDispersedDataId({'day_obs': 20200101, 'seq_num': 5}, butler)


class SimbadLocationTestCase(lsst.utils.tests.TestCase):
    def test_found(self):
        with mock.patch('astroquery.simbad.Simbad') as mockSimbad:
            mockSimbad.query_object.return_value = [
                {'RA': '02 00 00', 'DEC': '+10 00 00'}]
            loc = simbadLocationForTarget('HD 1')
            self.assertIsInstance(loc, geom.SpherePoint)

    def test_notFound(self):
        with mock.patch('astroquery.simbad.Simbad') as mockSimbad:
            mockSimbad.query_object.return_value = None
            with self.assertRaises(ValueError):
                simbadLocationForTarget('HD 1')

    def test_multipleFound(self):
        with mock.patch('astroquery.simbad.Simbad') as mockSimbad:
            mockSimbad.query_object.return_value = [
                {'RA': '02 00 00', 'DEC': '+10 00 00'},
                {'RA': '03 00 00', 'DEC': '+20 00 00'}]
            with self.assertRaises(ValueError):
                simbadLocationForTarget('HD 1')


class VizierLocationTestCase(lsst.utils.tests.TestCase):
    def _makeExpWithDate(self):
        exp = afwImage.ExposureF(30, 30)
        exp.setWcs(makeWcs())
        date = dafBase.DateTime(60000.0, dafBase.DateTime.MJD, dafBase.DateTime.TAI)
        exp.getInfo().setVisitInfo(afwImage.VisitInfo(date=date))
        return exp

    def test_found_withMotionCorrection(self):
        exp = self._makeExpWithDate()
        with mock.patch('astroquery.vizier.Vizier') as mockVizier:
            mockVizier.query_object.return_value = {'I/311/hip2': makeHipTable()}
            loc = vizierLocationForTarget(exp, 'HD 1', doMotionCorrection=True)
            self.assertIsInstance(loc, geom.SpherePoint)

    def test_found_noMotionCorrection(self):
        exp = self._makeExpWithDate()
        with mock.patch('astroquery.vizier.Vizier') as mockVizier:
            mockVizier.query_object.return_value = {'I/311/hip2': makeHipTable()}
            loc = vizierLocationForTarget(exp, 'HD 1', doMotionCorrection=False)
            self.assertIsInstance(loc, geom.SpherePoint)

    def test_notFound(self):
        exp = self._makeExpWithDate()
        with mock.patch('astroquery.vizier.Vizier') as mockVizier:
            # empty TableList -> indexing by str raises TypeError -> ValueError
            mockVizier.query_object.return_value = []
            with self.assertRaises(ValueError):
                vizierLocationForTarget(exp, 'HD 1', doMotionCorrection=False)


class GetTargetCentroidTestCase(lsst.utils.tests.TestCase):
    def _makeExpWithDate(self):
        exp = afwImage.ExposureF(30, 30)
        exp.setWcs(makeWcs())
        date = dafBase.DateTime(60000.0, dafBase.DateTime.MJD, dafBase.DateTime.TAI)
        exp.getInfo().setVisitInfo(afwImage.VisitInfo(date=date))
        return exp

    def test_vizierSuccess(self):
        exp = self._makeExpWithDate()
        with mock.patch('astroquery.vizier.Vizier') as mockVizier:
            mockVizier.query_object.return_value = {'I/311/hip2': makeHipTable()}
            pixCoord = getTargetCentroidFromWcs(exp, 'HD 1', doMotionCorrection=True)
            self.assertIsNotNone(pixCoord)

    def test_vizierFail_simbadSuccess(self):
        exp = self._makeExpWithDate()
        with mock.patch('astroquery.vizier.Vizier') as mockVizier, \
                mock.patch('astroquery.simbad.Simbad') as mockSimbad:
            mockVizier.query_object.return_value = []  # forces ValueError
            mockSimbad.query_object.return_value = [
                {'RA': '02 00 00', 'DEC': '+10 00 00'}]
            # doMotionCorrection True + simbad triggers the warning branch too
            pixCoord = getTargetCentroidFromWcs(exp, 'HD 1', doMotionCorrection=True)
            self.assertIsNotNone(pixCoord)

    def test_bothFail_returnsNone(self):
        exp = self._makeExpWithDate()
        with mock.patch('astroquery.vizier.Vizier') as mockVizier, \
                mock.patch('astroquery.simbad.Simbad') as mockSimbad:
            mockVizier.query_object.return_value = []
            mockSimbad.query_object.return_value = None  # ValueError in simbad
            pixCoord = getTargetCentroidFromWcs(exp, 'HD 1', doMotionCorrection=True)
            self.assertIsNone(pixCoord)

    def test_falsyTargetLocation_returnsNone(self):
        # Defensive branch: the lookup succeeds (no exception) but returns a
        # falsy location. Patch vizierLocationForTarget to return None.
        exp = self._makeExpWithDate()
        with mock.patch('lsst.atmospec.utils.vizierLocationForTarget',
                        return_value=None):
            pixCoord = getTargetCentroidFromWcs(exp, 'HD 1', doMotionCorrection=False)
            self.assertIsNone(pixCoord)

    def test_customLogger_noMotionCorrection(self):
        import logging
        exp = self._makeExpWithDate()
        logger = logging.getLogger('test.centroid')
        with mock.patch('astroquery.vizier.Vizier') as mockVizier:
            mockVizier.query_object.return_value = {'I/311/hip2': makeHipTable()}
            pixCoord = getTargetCentroidFromWcs(exp, 'HD 1',
                                                doMotionCorrection=False, logger=logger)
            self.assertIsNotNone(pixCoord)


class RunNotebookTestCase(lsst.utils.tests.TestCase):
    """Tests for runNotebook, with all butler/pipeline machinery mocked."""

    def _runWith(self, dataId, **kwargs):
        patchButler = mock.patch('lsst.atmospec.utils.dafButler')
        patchPipeline = mock.patch('lsst.atmospec.utils.Pipeline')
        patchExecutor = mock.patch('lsst.atmospec.utils.SeparablePipelineExecutor')
        patchDefaults = mock.patch('lsst.atmospec.utils.RegistryDefaults')
        with patchButler as mockDafButler, patchPipeline as mockPipeline, \
                patchExecutor as mockExecutor, patchDefaults:
            butlerInstance = mockDafButler.Butler.return_value
            butlerInstance.get.return_value = 'theSpectrum'
            pipelineInstance = mockPipeline.from_uri.return_value
            result = runNotebook(dataId, 'testOutputCollection', **kwargs)
            return result, butlerInstance, pipelineInstance, mockExecutor

    def test_basic(self):
        result, butler, pipeline, executor = self._runWith(
            {'day_obs': 20200101, 'seq_num': 5})
        self.assertEqual(result, 'theSpectrum')
        butler.get.assert_called_once()

    def test_dottedDataIdAndExtraInputs(self):
        result, butler, pipeline, executor = self._runWith(
            {'exposure.day_obs': 20200101, 'exposure.seq_num': 5},
            extraInputCollections='extra/collection', embargo=True)
        self.assertEqual(result, 'theSpectrum')

    def test_extraInputsList(self):
        result, butler, pipeline, executor = self._runWith(
            {'day_obs': 20200101, 'seq_num': 5},
            extraInputCollections=['extra/one', 'extra/two'])
        self.assertEqual(result, 'theSpectrum')

    def test_taskConfigs_connectionsAndPlainOption(self):
        # Build a real PipelineTaskConfig so ConnectionsConfigClass exists.
        # Iterating its .items() yields the connections config (exercising the
        # ConnectionsConfigClass isinstance branch and its inner loop) plus a
        # plain scalar field (exercising the addConfigOverride branch).
        import lsst.pipe.base as pipeBase
        import lsst.pipe.base.connectionTypes as cT

        class Connections(pipeBase.PipelineTaskConnections,
                          dimensions=("instrument",)):
            output = cT.Output(name="anOutput", doc="an output",
                               storageClass="Exposure",
                               dimensions=("instrument",))

        class MyConfig(pipeBase.PipelineTaskConfig,
                       pipelineConnections=Connections):
            val = pexConfig.Field(dtype=float, default=1.0, doc='val')

        config = MyConfig()
        result, butler, pipeline, executor = self._runWith(
            {'day_obs': 20200101, 'seq_num': 5},
            taskConfigs={'myTask': config})
        self.assertEqual(result, 'theSpectrum')
        # the plain 'val' option is applied
        appliedOptions = [c.args[1] for c in pipeline.addConfigOverride.call_args_list]
        self.assertIn('val', appliedOptions)

    def test_configOptions_plainAndConfigurable(self):
        class SubConfig(pexConfig.Config):
            x = pexConfig.Field(dtype=int, default=1, doc='x')

        class MainConfig(pexConfig.Config):
            sub = pexConfig.ConfigurableField(doc='c', target=object,
                                              ConfigClass=SubConfig)

        configurable = MainConfig().sub
        result, butler, pipeline, executor = self._runWith(
            {'day_obs': 20200101, 'seq_num': 5},
            configOptions={'myTask': {'threshold': 5.0,
                                      'configurableThing': configurable}})
        appliedOptions = [c.args[1] for c in pipeline.addConfigOverride.call_args_list]
        self.assertIn('threshold', appliedOptions)
        self.assertNotIn('configurableThing', appliedOptions)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
