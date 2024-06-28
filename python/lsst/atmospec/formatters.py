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

__all__ = ['SpectractorSpectrumFormatter',
           'SpectractorImageFormatter',
           'SpectractorFitParametersFormatter']

from typing import Any
from lsst.resources import ResourcePath
from lsst.daf.butler import FormatterV2
from spectractor.extractor.spectrum import Spectrum
from spectractor.extractor.images import Image
from spectractor.fit.fitter import read_fitparameter_json, write_fitparameter_json


class SpectractorSpectrumFormatter(FormatterV2):
    default_extension = '.fits'
    unsupported_parameters = None
    can_read_from_local_file = True

    def read_from_local_file(
        self, path: str, component: str | None = None, expected_size: int = -1
    ) -> Any:
        return Spectrum(path)

    def write_local_file(self, in_memory_dataset: Any, uri: ResourcePath) -> None:
        in_memory_dataset.save_spectrum(uri.ospath)


class SpectractorImageFormatter(FormatterV2):
    default_extension = '.fits'
    unsupported_parameters = None
    can_read_from_local_file = True

    def read_from_local_file(
        self, path: str, component: str | None = None, expected_size: int = -1
    ) -> Any:
        return Image(path)

    def write_local_file(self, in_memory_dataset: Any, uri: ResourcePath) -> None:
        in_memory_dataset.save_image(uri.ospath)


class SpectractorFitParametersFormatter(FormatterV2):
    default_extension = '.json'
    unsupported_parameters = None
    can_read_from_local_file = True

    def read_from_local_file(
        self, path: str, component: str | None = None, expected_size: int = -1
    ) -> Any:
        return read_fitparameter_json(path)

    def write_local_file(self, in_memory_dataset: Any, uri: ResourcePath) -> None:
        write_fitparameter_json(uri.ospath, in_memory_dataset)
