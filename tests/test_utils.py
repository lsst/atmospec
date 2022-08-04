#!/usr/bin/env python

#
# LSST Data Management System
#
# Copyright 2008-2017  AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
"""Test cases for atmospec."""

import unittest
import itertools
import numpy as np

import lsst.utils
import lsst.utils.tests
from lsst.atmospec.utils import argMaxNd, getSamplePoints, airMassFromRawMetadata


class AtmospecUtilsTestCase(lsst.utils.tests.TestCase):
    """A test case for atmospec."""

    def testImport(self):
        import lsst.atmospec.utils as utils  # noqa: F401

    def test_argMaxNd(self):
        data = np.ones((10, 10))

        data[2, 3] = 100
        data[1, 2] = -200
        maxLocation = argMaxNd(data)
        self.assertTrue(maxLocation == (2, 3))

        data3d = np.ones((10, 20, 15))
        data3d[3, 4, 5] = 2
        data3d[1, 2, 3] = -10
        maxLocation3d = argMaxNd(data3d)
        self.assertTrue(maxLocation3d == (3, 4, 5))

    def test_getSamplePoints(self):
        points = getSamplePoints(0, 100, 3, includeEndpoints=False, integers=False)
        self.assertEqual(points, [16.666666666666668, 50.0, 83.33333333333334])

        points = getSamplePoints(0, 100, 3, includeEndpoints=False, integers=True)
        self.assertEqual(points, [17, 50, 83])

        points = getSamplePoints(0, 100, 3, includeEndpoints=True, integers=False)
        self.assertEqual(points, [0, 50, 100])

        points = getSamplePoints(0, 100, 4, includeEndpoints=True, integers=False)
        self.assertEqual(points, [0, 33.333333333333336, 66.66666666666667, 100.0])

        points = getSamplePoints(0, 100, 4, includeEndpoints=False, integers=False)
        self.assertEqual(points, [12.5, 37.5, 62.5, 87.5])

        points = getSamplePoints(0, 100, 5, includeEndpoints=False, integers=False)
        self.assertEqual(points, [10.0, 30.0, 50.0, 70.0, 90.0])

        points = getSamplePoints(0, 100, 5, includeEndpoints=True, integers=False)
        self.assertEqual(points, [0, 25.0, 50.0, 75.0, 100.0])

        points = getSamplePoints(0, 100.1, 5, includeEndpoints=True, integers=False)
        self.assertEqual(points[-1], 100.1)

        points = getSamplePoints(0, 100.1, 5, includeEndpoints=True, integers=True)
        self.assertNotEqual(points[-1], 100.1)

        for ints in (True, False):
            with self.assertRaises(RuntimeError):
                getSamplePoints(0, 100, 1, includeEndpoints=True, integers=ints)

        for start, end in itertools.product((-1.5, -1, 0, 2.3), (0, 3.14, -1e9)):
            points = getSamplePoints(start, end, 2, includeEndpoints=True, integers=False)
            self.assertEqual(points, [start, end])

    def test_airmass(self):
        md = {
            # Minimalist header.
            "INSTRUME": "LATISS",
            "MJD-OBS": 60_000.0,
            "OBSID": "AT_O_20300101_00000",
            "AMSTART": 1.234,
        }
        self.assertEqual(airMassFromRawMetadata(md), 1.234)

        # Bad header should return 0.0.
        self.assertEqual(airMassFromRawMetadata({}), 0.0)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
