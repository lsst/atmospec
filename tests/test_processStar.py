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
"""Test cases for cp_pipe."""

from __future__ import absolute_import, division, print_function
import unittest

import lsst.utils
import lsst.utils.tests


class ProcessStarTestCase(lsst.utils.tests.TestCase):
    """A test case for atmospec."""

    def testImport(self):
        import lsst.atmospec as atmospec  # noqa: F401

    def testClassInstantiation(self):
        from lsst.atmospec.extraction import SpectralExtractionTask
        config = SpectralExtractionTask.ConfigClass
        task = SpectralExtractionTask(config=config)
        del config, task

    def test_calculateBackgroundMasking(self):
        from lsst.atmospec.extraction import SpectralExtractionTask
        import numpy as np
        import lsst.afw.image as afwImage

        task = SpectralExtractionTask()

        mi = afwImage.MaskedImageF(5, 5)
        mi.image.array[:] = np.ones((5, 5))

        nbins = 1

        bgImg = task._calculateBackground(mi, nbins)
        self.assertEqual(np.shape(mi.image.array), np.shape(bgImg.array))
        self.assertEqual(np.max(bgImg.array), 1.)

        mi.image.array[2, 2] = 100
        bgImg = task._calculateBackground(mi, nbins)
        self.assertTrue(np.max(bgImg.array) > 1.)

        MaskPixel = afwImage.MaskPixel
        mi.mask.array[2, 2] = afwImage.Mask[MaskPixel].getPlaneBitMask("DETECTED")
        bgImg = task._calculateBackground(mi, nbins)
        self.assertEqual(np.max(bgImg.array), 1.)

        mi.image.array[3, 3] = 200
        mi.mask.array[3, 3] = afwImage.Mask[MaskPixel].getPlaneBitMask("BAD")
        self.assertEqual(np.max(mi.image.array), 200)
        # don't include "BAD", but it's the default, so exclude explicitly
        bgImg = task._calculateBackground(mi, nbins, ignorePlanes="DETECTED")
        self.assertTrue(np.max(bgImg.array > 1.))

        # And now check the explicitly including it gets us back to where we were
        bgImg = task._calculateBackground(mi, nbins, ignorePlanes=["DETECTED", "BAD"])
        self.assertEqual(np.max(bgImg.array), 1)

    def test_getSamplePoints(self):
        from lsst.atmospec.extraction import getSamplePoints
        import itertools

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


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
