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

import logging
import numpy as np


__all__ = ['DispersionRelation']


class DispersionRelation:
    def __init__(self, observedLines, spectralLines):
        """The dispersion relation, relating pixel to wavelength and vice versa

        Parameters:
        -----------
        observedLines : `list` of `float`
            The central wavelength of the observed lines in the spectrum in pix

        spectralLines : `list` of `float`
            The central wavelength of the spectral lines present in the source
            in nm.

        Notes:
        ------
        The current implementation just supplies linear transformations
        but future extensions can support higher order polynomials,
        spline-fits, distortions etc
        """
        self.observedLines = observedLines
        self.spectralLines = spectralLines

        self.log = logging.getLogger('lsst.atmospec.dispersionRelation')
        self.pix2wlCoeffs = self._calcCoefficients()

    def _calcCoefficients(self):
        if((self.observedLines is None) or (self.spectralLines is None)):
            self.log.warn('Missing input for _calcCoefficients, default transformation: 1 to 1 ')
            self.observedLines = [1, 2]
            self.spectralLines = [1, 2]
        pix2wlCoeffs = np.polyfit(self.observedLines, self.spectralLines, deg=1)

        # xxx change to debug
        self.log.info('Pixel -> Wavelength linear transformation coefficients : ' + str(pix2wlCoeffs))
        return pix2wlCoeffs

    def Wavelength2Pixel(self, wavelength):
        """Currently just a linear transform"""
        wavelength = np.asarray(wavelength, dtype=np.float64)
        coef = self.pix2wlCoeffs
        return (wavelength-coef[1]) / coef[0]

    def Pixel2Wavelength(self, pixel):
        """Currently just a linear transform"""
        pixel = np.asarray(pixel, dtype=np.float64)
        coef = self.pix2wlCoeffs
        return np.array(coef[1] + coef[0] * pixel)
