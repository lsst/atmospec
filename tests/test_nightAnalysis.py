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
"""Test cases for atmospec.nightAnalysis."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from astropy.table import Table

import lsst.utils
import lsst.utils.tests

import lsst.atmospec.nightAnalysis as nightAnalysis
from lsst.atmospec.nightAnalysis import (
    NightStellarSpectra,
    getLineValue,
    _getRowNum,
    LINE_NAMES,
)


def makeStrTable():
    """Build a str-keyed table for the free-function tests.

    Row 0 is deliberately a valid line so that the ``not rowNum``
    branch (rowNum == 0 is falsy) can be exercised.
    """
    table = Table()
    table["name"] = [LINE_NAMES["O2"], LINE_NAMES["H_alpha"], LINE_NAMES["H_beta"]]
    table["value"] = [10.0, 20.0, 30.0]
    return table


def makeByteTable(lineNames):
    """Build a byte-string keyed table like a real extraction file."""
    table = Table()
    encoded = np.array([LINE_NAMES[name].encode("utf-8") for name in lineNames])
    table["name"] = encoded
    table["value"] = np.arange(len(lineNames), dtype=float)
    return table


class FakeQuery:
    """Iterable stand-in for a butler query result with order_by."""

    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def order_by(self, *args, **kwargs):
        return self


def makeRecords():
    """Exposure records covering the seq_nums used in the tests."""
    records = []
    for seqNum, physFilter, mjd, za in (
        (1, "filtA~disp1", 60000.1, 20.0),
        (2, "filtB~disp2", 60000.2, 40.0),
        (3, "filtA~disp1", 60000.3, 25.0),
        (4, "filtB~disp2", 60000.4, 30.0),
    ):
        records.append(SimpleNamespace(
            seq_num=seqNum,
            physical_filter=physFilter,
            zenith_angle=za,
            timespan=SimpleNamespace(begin=SimpleNamespace(mjd=mjd)),
        ))
    return records


def makeButlerClass(records, extractionMap):
    """Create a fake Butler class.

    ``extractionMap`` maps seq_num -> table, ``None`` (returns None) or
    the string ``"lookup"`` (raises LookupError inside get).
    """

    class FakeButler:
        def __init__(self, *args, **kwargs):
            self.registry = SimpleNamespace(
                queryDimensionRecords=lambda *a, **k: FakeQuery(records))

        def get(self, datasetType, *, seq_num, day_obs):
            result = extractionMap.get(seq_num, None)
            if result == "lookup":
                raise LookupError("no such dataset")
            return result

    return FakeButler


def makeInstance(*, dayObs=20200101, target="HD12345", ignoreSeqNums=[],
                 records=None, extractionMap=None, dispersedSeqNums=(1, 2, 3),
                 passInstance=False):
    """Construct a NightStellarSpectra with a fully mocked butler."""
    if records is None:
        records = makeRecords()
    if extractionMap is None:
        extractionMap = {1: makeByteTable(["H_alpha", "H_beta"]),
                         2: makeByteTable(["H_alpha", "H_gamma"])}

    FakeButler = makeButlerClass(records, extractionMap)

    def fakeDispersed(dataId, butler):
        return dataId["seq_num"] in dispersedSeqNums

    butlerInput = FakeButler() if passInstance else object()

    with patch.object(nightAnalysis.dafButler, "Butler", FakeButler), \
            patch.object(nightAnalysis, "isDispersedDataId", side_effect=fakeDispersed):
        instance = NightStellarSpectra(butlerInput, dayObs, target,
                                       ignoreSeqNums=ignoreSeqNums)
    return instance


class NightAnalysisFunctionsTestCase(lsst.utils.tests.TestCase):
    """Tests for the free functions _getRowNum and getLineValue."""

    def test_getRowNum_found(self):
        table = makeStrTable()
        self.assertEqual(_getRowNum(table, "O2"), 0)
        self.assertEqual(_getRowNum(table, "H_alpha"), 1)
        self.assertEqual(_getRowNum(table, "H_beta"), 2)

    def test_getRowNum_missingLine(self):
        table = makeStrTable()
        # Valid line name but not present in the table -> None.
        self.assertIsNone(_getRowNum(table, "water"))

    def test_getRowNum_unknownName(self):
        table = makeStrTable()
        with self.assertRaises(RuntimeError):
            _getRowNum(table, "not_a_real_line")

    def test_getLineValue_found(self):
        table = makeStrTable()
        # H_alpha is at row 1 (truthy) so the real value is returned.
        self.assertEqual(getLineValue(table, "H_alpha", "value"), 20.0)

    def test_getLineValue_rowZeroTreatedAsMissing(self):
        table = makeStrTable()
        # O2 is at row 0 which is falsy, so it is treated as missing.
        self.assertTrue(np.isnan(getLineValue(table, "O2", "value")))
        with self.assertRaises(ValueError):
            getLineValue(table, "O2", "value", nanForMissingValues=False)

    def test_getLineValue_missingNan(self):
        table = makeStrTable()
        self.assertTrue(np.isnan(getLineValue(table, "water", "value")))

    def test_getLineValue_missingRaises(self):
        table = makeStrTable()
        with self.assertRaises(ValueError):
            getLineValue(table, "water", "value", nanForMissingValues=False)


class NightStellarSpectraTestCase(lsst.utils.tests.TestCase):
    """Tests for the NightStellarSpectra class."""

    def test_construction_elseBranch(self):
        night = makeInstance(ignoreSeqNums=[3])
        # 1, 2 dispersed and kept; 3 dispersed but ignored; 4 not dispersed.
        self.assertEqual(night.seqNums, [1, 2])
        self.assertEqual(set(night.data.keys()), {1, 2})
        self.assertEqual(night.dayObs, 20200101)
        self.assertEqual(night.targetName, "HD12345")

    def test_construction_isinstanceBranch(self):
        # Passing an actual FakeButler instance exercises the isinstance
        # True branch of __init__.
        night = makeInstance(passInstance=True)
        self.assertEqual(night.seqNums, [1, 2])

    def test_construction_missingExtraction(self):
        # seq_num 2 raises LookupError inside get -> not loaded.
        extractionMap = {1: makeByteTable(["H_alpha"]), 2: "lookup"}
        night = makeInstance(extractionMap=extractionMap)
        self.assertEqual(night.seqNums, [1])
        self.assertEqual(set(night.data.keys()), {1})

    def test_construction_noneExtraction(self):
        # get returns None -> falsy -> not loaded.
        extractionMap = {1: makeByteTable(["H_alpha"]), 2: None}
        night = makeInstance(extractionMap=extractionMap)
        self.assertEqual(night.seqNums, [1])

    def test_isDispersed(self):
        night = makeInstance()
        with patch.object(nightAnalysis, "isDispersedDataId",
                          side_effect=lambda dataId, butler: dataId["seq_num"] == 5):
            self.assertTrue(night.isDispersed(5))
            self.assertFalse(night.isDispersed(6))

    def test_readOneExtractionFile_lookupError(self):
        night = makeInstance()
        with patch.object(night.butler, "get", side_effect=LookupError):
            self.assertIsNone(night._readOneExtractionFile(999))

    def test_removeSeqNums(self):
        night = makeInstance(ignoreSeqNums=[3])
        night.removeSeqNums([2, 42])  # 42 not present -> no-op
        self.assertEqual(night.seqNums, [1])
        self.assertNotIn(2, night.data)

    def test_getFilterDisperserSet(self):
        night = makeInstance(ignoreSeqNums=[3])
        self.assertEqual(night.getFilterDisperserSet(),
                         {"filtA~disp1", "filtB~disp2"})

    def test_getFilterSet(self):
        night = makeInstance(ignoreSeqNums=[3])
        self.assertEqual(night.getFilterSet(), {"filtA", "filtB"})

    def test_getDisperserSet(self):
        night = makeInstance(ignoreSeqNums=[3])
        self.assertEqual(night.getDisperserSet(), {"disp1", "disp2"})

    def test_getLineValue_instanceMethod(self):
        night = makeInstance(ignoreSeqNums=[3])
        # Byte-string tables never match the str line names, so this is nan.
        self.assertTrue(np.isnan(night.getLineValue(1, "H_alpha", "value")))

    def test_getLineValues(self):
        night = makeInstance(ignoreSeqNums=[3])
        # Add a seq_num that is not in data to exercise the else branch.
        night.seqNums = [1, 2, 99]
        values = night.getLineValues("H_alpha", "value")
        self.assertEqual(len(values), 3)
        self.assertTrue(all(np.isnan(v) for v in values))

    def test_getAllTableLines_intermittent(self):
        night = makeInstance(ignoreSeqNums=[3])
        lines = night.getAllTableLines(includeIntermittentLines=True)
        self.assertEqual(set(lines), {"H_alpha", "H_beta", "H_gamma"})

    def test_getAllTableLines_common(self):
        night = makeInstance(ignoreSeqNums=[3])
        lines = night.getAllTableLines(includeIntermittentLines=False)
        self.assertEqual(set(lines), {"H_alpha"})

    def test_getObsTimes(self):
        night = makeInstance(ignoreSeqNums=[3])
        self.assertEqual(night.getObsTimes(), [60000.1, 60000.2])

    def test_getAirmasses(self):
        night = makeInstance(ignoreSeqNums=[3])
        airmasses = night.getAirmasses()
        self.assertEqual(len(airmasses), 2)
        expected = 1.0 / np.cos(np.deg2rad(20.0))
        self.assertFloatsAlmostEqual(airmasses[0], expected)

    def test_printObservationTable(self):
        night = makeInstance(ignoreSeqNums=[3])
        # Simply exercise the code path.
        night.printObservationTable()


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
