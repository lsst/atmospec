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
"""Test cases for atmospec dispersion relation."""

import unittest
import numpy as np

import lsst.utils
import lsst.utils.tests
from lsst.atmospec.dispersion import DispersionRelation


class DispersionRelationTestCase(lsst.utils.tests.TestCase):
    """A test case for the DispersionRelation class."""

    def testImport(self):
        import lsst.atmospec.dispersion as dispersion  # noqa: F401

    def test_init_and_coefficients(self):
        # A simple 1-to-1 mapping: pixel == wavelength
        observedLines = [1.0, 2.0, 3.0]
        spectralLines = [1.0, 2.0, 3.0]
        dr = DispersionRelation(observedLines, spectralLines)
        self.assertEqual(dr.observedLines, observedLines)
        self.assertEqual(dr.spectralLines, spectralLines)
        self.assertEqual(len(dr.pix2wlCoeffs), 2)
        # slope ~ 1, intercept ~ 0
        self.assertAlmostEqual(dr.pix2wlCoeffs[0], 1.0)
        self.assertAlmostEqual(dr.pix2wlCoeffs[1], 0.0)

    def test_init_nontrivial(self):
        # wavelength = 2 * pixel + 5
        observedLines = [0.0, 10.0, 20.0]
        spectralLines = [5.0, 25.0, 45.0]
        dr = DispersionRelation(observedLines, spectralLines)
        self.assertAlmostEqual(dr.pix2wlCoeffs[0], 2.0)
        self.assertAlmostEqual(dr.pix2wlCoeffs[1], 5.0)

    def test_calcCoefficients_none_observed(self):
        # observedLines is None -> warning branch, defaults to 1-to-1
        dr = DispersionRelation(None, [1.0, 2.0])
        self.assertEqual(dr.observedLines, [1, 2])
        self.assertEqual(dr.spectralLines, [1, 2])
        self.assertAlmostEqual(dr.pix2wlCoeffs[0], 1.0)
        self.assertAlmostEqual(dr.pix2wlCoeffs[1], 0.0)

    def test_calcCoefficients_none_spectral(self):
        # spectralLines is None -> warning branch, defaults to 1-to-1
        dr = DispersionRelation([1.0, 2.0], None)
        self.assertEqual(dr.observedLines, [1, 2])
        self.assertEqual(dr.spectralLines, [1, 2])
        self.assertAlmostEqual(dr.pix2wlCoeffs[0], 1.0)
        self.assertAlmostEqual(dr.pix2wlCoeffs[1], 0.0)

    def test_calcCoefficients_both_none(self):
        dr = DispersionRelation(None, None)
        self.assertEqual(dr.observedLines, [1, 2])
        self.assertEqual(dr.spectralLines, [1, 2])

    def test_Pixel2Wavelength_scalar(self):
        # wavelength = 2 * pixel + 5
        dr = DispersionRelation([0.0, 10.0], [5.0, 25.0])
        wl = dr.Pixel2Wavelength(3.0)
        self.assertAlmostEqual(float(wl), 11.0)

    def test_Pixel2Wavelength_array(self):
        dr = DispersionRelation([0.0, 10.0], [5.0, 25.0])
        pixels = np.array([0.0, 1.0, 2.0])
        wl = dr.Pixel2Wavelength(pixels)
        np.testing.assert_allclose(wl, [5.0, 7.0, 9.0])

    def test_Wavelength2Pixel_scalar(self):
        dr = DispersionRelation([0.0, 10.0], [5.0, 25.0])
        pix = dr.Wavelength2Pixel(11.0)
        self.assertAlmostEqual(float(pix), 3.0)

    def test_Wavelength2Pixel_array(self):
        dr = DispersionRelation([0.0, 10.0], [5.0, 25.0])
        wavelengths = np.array([5.0, 7.0, 9.0])
        pix = dr.Wavelength2Pixel(wavelengths)
        np.testing.assert_allclose(pix, [0.0, 1.0, 2.0], atol=1e-9)

    def test_roundTrip_scalar(self):
        dr = DispersionRelation([0.0, 10.0, 20.0], [5.0, 25.0, 45.0])
        pixel = 7.0
        wl = dr.Pixel2Wavelength(pixel)
        recovered = dr.Wavelength2Pixel(wl)
        self.assertAlmostEqual(float(recovered), pixel)

    def test_roundTrip_array(self):
        dr = DispersionRelation([0.0, 10.0, 20.0], [5.0, 25.0, 45.0])
        pixels = np.array([1.0, 5.0, 12.5, 100.0])
        wl = dr.Pixel2Wavelength(pixels)
        recovered = dr.Wavelength2Pixel(wl)
        np.testing.assert_allclose(recovered, pixels)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
