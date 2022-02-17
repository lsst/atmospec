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

__all__ = ['SpectractorSpectrumFormatter', 'SpectractorImageFormatter']

from lsst.daf.butler.formatters.file import FileFormatter
from lsst.daf.butler.formatters.pickle import PickleFormatter
from spectractor.extractor.spectrum import Spectrum
from spectractor.extractor.images import Image


class SpectractorSpectrumFormatter(FileFormatter):
    extension = '.fits'
    unsupportedParameters = None

    def _readFile(self, path, pytype=None):
        return Spectrum(path)

    def _writeFile(self, inMemoryDataset):
        inMemoryDataset.save_spectrum(self.fileDescriptor.location.path)


class SpectractorImageFormatter(FileFormatter):
    extension = '.fits'
    unsupportedParameters = None

    def _readFile(self, path, pytype=None):
        return Image(path)

    def _writeFile(self, inMemoryDataset):
        inMemoryDataset.save_image(self.fileDescriptor.location.path)


class SpectractionFormatter(PickleFormatter):
    pass
