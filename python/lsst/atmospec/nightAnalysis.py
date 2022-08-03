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

from astropy.coordinates import Angle
import numpy as np

import lsst.daf.butler as dafButler
from lsst.obs.lsst.translators.lsst import FILTER_DELIMITER
from .utils import isDispersedDataId

__all__ = ["NightStellarSpectra", "getLineValue", "LINE_NAMES"]

LINE_NAME_COLUMN = 0

# mapping from the table's internal naming to something you can type
LINE_NAMES = {'H_alpha': '$H\\alpha$',
              'H_beta': '$H\\beta$',
              'H_gamma': '$H\\gamma$',
              'H_delta': '$H\\delta$',
              'H_epsilon': '$H\\epsilon',
              'O2_b': '$O_2(B)$',
              'O2_z': '$O_2(Z)$',
              'O2_y': '$O_2(Y)$',
              'O2': '$O_2$',
              'water': '$H_2 O$',  # this actually maps to more than one line! Needs fixing upstream
              }

LINE_NAMES_REVERSED = {v: k for k, v in LINE_NAMES.items()}


def _getRowNum(table, lineName):
    """Surely there's a better/quicker way to do this"""
    try:
        actualName = LINE_NAMES[lineName]
    except KeyError:
        raise RuntimeError(f"Unknown line name {lineName}") from None  # error chaining unhelpful here

    for rowNum, row in enumerate(table.iterrows()):
        if row[LINE_NAME_COLUMN] == actualName:
            return rowNum
    # NB: this function is not combined with getLineValue() because
    # the None needs to checked for, as dereferencing the table with
    # t[colName][None] returns column itself
    return None


def getLineValue(table, lineName, columnName, nanForMissingValues=True):
    """Surely there's a better/quicker way to do this"""

    rowNum = _getRowNum(table, lineName)
    if not rowNum:
        if nanForMissingValues:
            return np.nan
        raise ValueError(f"Line {lineName} not found in table")

    return table[columnName][rowNum]


class NightStellarSpectra:
    """Class for holding the results for a night's observations of a given star
    """

    def __init__(self, butler, dayObs, targetName, *, instrument="LATISS", ignoreSeqNums=[],
                 collections=None):
        self.dayObs = int(dayObs)
        self.targetName = targetName

        if isinstance(butler, dafButler.Butler):
            self.butler = dafButler.Butler(butler=butler, instrument=instrument, collections=collections)
        else:
            self.butler = dafButler.Butler(butler, instrument=instrument, collections=collections)
        self._loadExtractions(ignoreSeqNums)
        # xxx maybe just load everything and then call removeSeqNums()?

    def isDispersed(self, seqNum):
        """Check if this observation is dispersed."""
        dataId = {"day_obs": self.dayObs, "seq_num": seqNum}
        return isDispersedDataId(dataId, self.butler)

    def _readOneExtractionFile(self, seqNum):
        datasetType = "extraction"
        try:
            return self.butler.get(datasetType,
                                   seq_num=seqNum,
                                   day_obs=self.dayObs)
        except LookupError:
            return None

    def _loadExtractions(self, ignoreSeqNums):
        # Get all the observations for the night.
        where = "exposure.day_obs = dayObs and exposure.target_name = targetName"
        records = self.butler.registry.queryDimensionRecords("exposure",
                                                             where=where,
                                                             bind={"dayObs": self.dayObs,
                                                                   "targetName": self.targetName},
                                                             )
        allSeqNums = [r.seq_num for r in records]
        self.seqNums = []
        nIgnored = 0

        for seqNum in allSeqNums:
            if self.isDispersed(seqNum):
                if seqNum not in ignoreSeqNums:
                    self.seqNums.append(seqNum)
                else:
                    nIgnored += 1

        msg = (f"Found {len(self.seqNums)+nIgnored} dispersed observations of "
               + f"{self.targetName} on {self.dayObs} in registry")
        if nIgnored:
            msg += f" of which {nIgnored} were skipped."
        print(msg)

        self.seqNums = sorted(self.seqNums)  # not guaranteed to be in order, I don't think

        self.data = {}
        successes = []
        for seqNum in self.seqNums:
            linesTable = self._readOneExtractionFile(seqNum)
            if linesTable:
                self.data[seqNum] = linesTable
                successes.append(seqNum)
        self.seqNums = successes
        print(f"Loaded extractions for {len(self.data.keys())} of the above")
        return

    def removeSeqNums(self, seqNums):
        for seqNum in seqNums:
            if seqNum in self.seqNums:
                assert seqNum in self.data.keys()
                self.seqNums.remove(seqNum)
                self.data.pop(seqNum)

    def _getExposureRecords(self):
        """Retrieve the exposure records for the relevant exposures.

        Returned in same order as ``self.seqNums``.
        """
        where = "exposure.day_obs = dayObs"
        records = self.butler.registry.queryDimensionRecords("exposure",
                                                             where=where,
                                                             bind={"dayObs": self.dayObs},
                                                             )
        seqNums = set(self.seqNums)  # Use set for faster lookup.

        # The order is random, but we are required to return it in the
        # original order.
        recordsBySeqNum = {r.seq_num: r for r in records if r.seq_num in seqNums}

        # There should be an entry for ever seqNum since the seqNum list
        # has already been pre-filtered.
        return [recordsBySeqNum[seqNum] for seqNum in self.seqNums]

    def getFilterDisperserSet(self):
        # Doing a query per seqNum is going to be slow, so query for the
        # whole night and filter.
        records = self._getExposureRecords()
        fullFilters = {r.physical_filter for r in records}
        return fullFilters

    def getFilterSet(self):
        fullFilters = self.getFilterDisperserSet()
        return {filt.split(FILTER_DELIMITER)[0] for filt in fullFilters}

    def getDisperserSet(self):
        fullFilters = self.getFilterDisperserSet()
        return {filt.split(FILTER_DELIMITER)[1] for filt in fullFilters}

    def getLineValue(self, seqNum, lineName, columnName):
        # just convenient to be able to call this on a class instance as well
        # as the free floating function - nothing deep happening here
        return getLineValue(self.data[seqNum], lineName, columnName)

    def getLineValues(self, lineName, columnName):
        return [getLineValue(self.data[seqNum], lineName, columnName) if seqNum in self.data else np.nan
                for seqNum in self.seqNums]

    def getAllTableLines(self, includeIntermittentLines=True):
        lineSets = []
        for seqNum, table in self.data.items():
            lineSets.append(set(table.columns[LINE_NAME_COLUMN].data))

        if includeIntermittentLines:
            ret = set.union(*lineSets)
        else:
            ret = set.intersection(*lineSets)

        # we get byte strings back, so translate, and also lookup in
        # line name dictionary for convenience
        lines = [text.decode('utf-8') for text in ret]
        lineNames = [LINE_NAMES_REVERSED[line] for line in lines]
        return lineNames

    def getObsTimes(self):
        # TODO: Add option to subtract int part and multiply up
        # to make it a human-readable small number
        records = self._getExposureRecords()
        dates = [r.timespan.begin for r in records]
        return [d.mjd for d in dates]

    def getAirmasses(self):
        records = self._getExposureRecords()
        angles = [Angle(r.zenith_angle, unit="deg") for r in records]
        return [1/np.cos(za.radian) for za in angles]

    def printObservationTable(self):
        records = self._getExposureRecords()
        for r in records:
            print(r.seq_num, r.physical_filter)
