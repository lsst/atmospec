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

import os

from lsst.daf.butler.formatters.file import FileFormatter
from spectractor.extractor.spectrum import Spectrum
from spectractor.extractor.images import Image


class SpectractorSpectrumFormatter(FileFormatter):
    extension = '.fits'
    unsupportedParameters = None

    def _readFile(self, path, pytype=None):
        spectrogramPath = path + 'spectrogram.fits'
        psfPath = path + 'table.csv'
        # TODO: Add round-tripping of detected lines with lines.csv once
        # Spectractor supports that
        if os.path.isfile(spectrogramPath) and os.path.isfile(psfPath):
            return Spectrum(path, fast_load=False,
                            spectrogram_file_name_override=spectrogramPath,
                            psf_file_name_override=psfPath)
        else:
            return Spectrum(path, fast_load=True)

    def _writeFile(self, inMemoryDataset):
        spectrogramFilename = self.fileDescriptor.location.path + 'spectrogram.fits'
        linesFilename = self.fileDescriptor.location.path + 'lines.csv'
        psfFilename = self.fileDescriptor.location.path + 'table.csv'

        inMemoryDataset.save_spectrum(self.fileDescriptor.location.path)
        inMemoryDataset.save_spectrogram(spectrogramFilename, overwrite=True)
        inMemoryDataset.lines.print_detected_lines(output_file_name=linesFilename,
                                                   overwrite=True, amplitude_units=inMemoryDataset.units)
        inMemoryDataset.chromatic_psf.table.write(psfFilename, overwrite=True)


class SpectractorImageFormatter(FileFormatter):
    extension = '.fits'
    unsupportedParameters = None

    def _readFile(self, path, pytype=None):
        return Image(path)

    def _writeFile(self, inMemoryDataset):
        inMemoryDataset.save_image(self.fileDescriptor.location.path)
