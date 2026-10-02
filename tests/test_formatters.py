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
"""Test cases for atmospec Spectractor formatters."""

import unittest
from unittest import mock

import lsst.utils
import lsst.utils.tests
from lsst.atmospec.formatters import (
    SpectractorSpectrumFormatter,
    SpectractorImageFormatter,
    SpectractorFitParametersFormatter,
)


class FormattersTestCase(lsst.utils.tests.TestCase):
    """A test case for the Spectractor formatters.

    The formatter methods do not depend on any instance state, so instances
    are created with ``object.__new__`` to avoid the ``FormatterV2``
    constructor, which requires a full ``FileDescriptor``/``DatasetRef``.
    """

    def testImport(self):
        import lsst.atmospec.formatters as formatters  # noqa: F401

    def test_classAttributes(self):
        for cls, ext in (
            (SpectractorSpectrumFormatter, '.fits'),
            (SpectractorImageFormatter, '.fits'),
            (SpectractorFitParametersFormatter, '.json'),
        ):
            self.assertEqual(cls.default_extension, ext)
            self.assertIsNone(cls.unsupported_parameters)
            self.assertTrue(cls.can_read_from_local_file)

    def test_spectrum_read_from_local_file(self):
        formatter = object.__new__(SpectractorSpectrumFormatter)
        sentinel = mock.MagicMock(name="Spectrum instance")
        with mock.patch("lsst.atmospec.formatters.Spectrum",
                        return_value=sentinel) as mockSpectrum:
            result = formatter.read_from_local_file("/some/path.fits")
        mockSpectrum.assert_called_once_with("/some/path.fits")
        self.assertIs(result, sentinel)

    def test_spectrum_write_local_file(self):
        formatter = object.__new__(SpectractorSpectrumFormatter)
        dataset = mock.MagicMock(name="Spectrum dataset")
        uri = mock.MagicMock()
        uri.ospath = "/out/path.fits"
        formatter.write_local_file(dataset, uri)
        dataset.save_spectrum.assert_called_once_with("/out/path.fits")

    def test_image_read_from_local_file(self):
        formatter = object.__new__(SpectractorImageFormatter)
        sentinel = mock.MagicMock(name="Image instance")
        with mock.patch("lsst.atmospec.formatters.Image",
                        return_value=sentinel) as mockImage:
            result = formatter.read_from_local_file("/some/path.fits")
        mockImage.assert_called_once_with("/some/path.fits")
        self.assertIs(result, sentinel)

    def test_image_write_local_file(self):
        formatter = object.__new__(SpectractorImageFormatter)
        dataset = mock.MagicMock(name="Image dataset")
        uri = mock.MagicMock()
        uri.ospath = "/out/image.fits"
        formatter.write_local_file(dataset, uri)
        dataset.save_image.assert_called_once_with("/out/image.fits")

    def test_fitparameters_read_from_local_file(self):
        formatter = object.__new__(SpectractorFitParametersFormatter)
        sentinel = mock.MagicMock(name="fit parameters")
        with mock.patch("lsst.atmospec.formatters.read_fitparameter_json",
                        return_value=sentinel) as mockRead:
            result = formatter.read_from_local_file("/some/params.json")
        mockRead.assert_called_once_with("/some/params.json")
        self.assertIs(result, sentinel)

    def test_fitparameters_write_local_file(self):
        formatter = object.__new__(SpectractorFitParametersFormatter)
        dataset = mock.MagicMock(name="fit parameters dataset")
        uri = mock.MagicMock()
        uri.ospath = "/out/params.json"
        with mock.patch("lsst.atmospec.formatters.write_fitparameter_json") as mockWrite:
            formatter.write_local_file(dataset, uri)
        mockWrite.assert_called_once_with("/out/params.json", dataset)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
