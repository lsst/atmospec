#
# LSST Data Management System
#
# Copyright 2008-2018  AURA/LSST.
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

import copy
import numpy as np


def makeGainFlat(detectorExposure, gainDict):
    detector = detectorExposure.getDetector()
    ampNames = set(list(a.getName() for a in detector))
    assert set(gainDict.keys()) == ampNames

    flat = copy.copy(detectorExposure)
    for amp in detector:
        bbox = amp.getBBox()
        flat[bbox].maskedImage.image.array[:, :] = gainDict[amp.getName()]
    flat.maskedImage.mask[:] = 0x0
    flat.maskedImage.variance[:] = 0.0

    return flat


def argMaxNd(array):
    """Get the index of the max value of an array.

    If there are multiple occurences of the maximum value
    just return the first.
    """
    return np.unravel_index(np.argmax(array, axis=None), array.shape)


def getSamplePoints(start, stop, nSamples, includeEndpoints=False, integers=False):
    """Get the locations of the coordinates to use to sample a range evenly

    Divide a range up and return the coordinated to use in order to evenly
    sample the range. If asking for integers, rounded values are returned,
    rather than int-truncated ones.

    If not including the endpoints, divide the (stop-start) range into nSamples,
    and return the midpoint of each section, thus leaving a sectionLength/2 gap
    between the first/last samples and the range start/end.

    If including the endpoints, the first and last points will be
    start, stop respectively, and other points will be the endpoints of the
    remaining nSamples-1 sections.

    Visually, for a range:

    |--*--|--*--|--*--|--*--| return * if not including end points, n=4
    |-*-|-*-|-*-|-*-|-*-|-*-| return * if not including end points, n=6

    *-----*-----*-----*-----* return * if we ARE including end points, n=4
    *---*---*---*---*---*---* return * if we ARE including end points, n=6
    """

    if not includeEndpoints:
        r = (stop-start)/(2*nSamples)
        points = [((2*pointNum+1)*r) for pointNum in range(nSamples)]
    else:
        if nSamples <= 1:
            raise RuntimeError('nSamples must be >= 2 if including endpoints')
        if nSamples == 2:
            points = [start, stop]
        else:
            r = (stop-start)/(nSamples-1)
            points = [start]
            points.extend([((pointNum)*r) for pointNum in range(1, nSamples)])

    if integers:
        return [int(x) for x in np.round(points)]
    return points
