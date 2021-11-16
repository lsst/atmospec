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
import numpy as np

import lsst.utils
import lsst.utils.tests
import lsst.afw.image as afwImage
from lsst.atmospec.extraction import SpectralExtractionTask

TRACEBACKS_FOR_ABI_WARNINGS = False

if TRACEBACKS_FOR_ABI_WARNINGS:
    import traceback
    import warnings
    import sys

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

        log = file if hasattr(file, 'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback


class ProcessStarTestCase(lsst.utils.tests.TestCase):
    """A test case for atmospec."""

    def testImport(self):
        import lsst.atmospec as atmospec  # noqa: F401

    def testClassInstantiation(self):
        config = SpectralExtractionTask.ConfigClass
        task = SpectralExtractionTask(config=config)
        del config, task

    def test_calculateBackgroundMasking(self):
        task = SpectralExtractionTask()

        for nbins in [1, 2, 3]:

            mi = afwImage.MaskedImageF(5, 5)
            mi.image.array[:] = np.ones((5, 5))

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

            # And now check thet explicitly including it gets us back
            # to where we were
            bgImg = task._calculateBackground(mi, nbins, ignorePlanes=["DETECTED", "BAD"])
            self.assertEqual(np.max(bgImg.array), 1)

        for nbins in [5, 15, 50]:
            mi = afwImage.MaskedImageF(5, 5)
            mi.image.array[:] = np.ones((5, 5))

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

            # don't include "BAD", but it's the default
            # so exclude explicitly
            bgImg = task._calculateBackground(mi, nbins, ignorePlanes="DETECTED")
            self.assertTrue(np.max(bgImg.array > 1.))

            # And now check that explicitly including it gets us back
            # to where we were
            bgImg = task._calculateBackground(mi, nbins, ignorePlanes=["DETECTED", "BAD"])
            self.assertEqual(np.max(bgImg.array), 1)

        # TODO:
        # should make a new test that makes a larger image and actually tests
        # interpolation and background calculations
        # also, I'm sure this could be tidied up with itertools


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
