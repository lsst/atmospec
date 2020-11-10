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
# This program is distributed in the hope hat it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#

import os
from astropy.table import Table
from astropy.time import Time
import numpy as np

import lsst.daf.persistence as dafPersist
import lsst.daf.persistence.butlerExceptions as butlerExceptions
from lsst.obs.lsst.translators.lsst import FILTER_DELIMITER

from .utils import airMassFromRawMetadata

__all__ = ["NightStellarSpectra", "getLineValue"]

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
              'fakeLine': 'does_not_exist'}

LINE_NAMES_REVERSED = {v: k for k, v in LINE_NAMES.items()}


def _getRowNum(table, lineName):
    """Surely there's a better/quicker way to do this"""
    try:
        actualName = LINE_NAMES[lineName]
    except KeyError:
        raise RuntimeError(f"Unknown line name {lineName}") from None  # chaining unhelpful here

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


class NightStellarSpectra():
    """Class for holding the results for a night's observations of a given star
    """

    def __init__(self, rerunDir, dayObs, targetName, *, butler=None):
        self.rerunDir = rerunDir
        self.dayObs = dayObs
        self.targetName = targetName
        if butler:
            self.butler = butler
        else:
            self.butler = dafPersist.Butler(rerunDir)
        self._loadExtractions()

    def isDispersed(self, seqNum):
        """Match object and check is dispersed"""
        filt = self.butler.queryMetadata('raw', 'filter', dayObs=self.dayObs, seqNum=seqNum)[0]
        grating = filt.split(FILTER_DELIMITER)[1]
        if "ronchi" in grating:
            return True
        return False

    def _expIdFromSeqNum(self, seqNum):
        return self.butler.queryMetadata('raw', 'expId', dayObs=self.dayObs, seqNum=seqNum)[0]

    def _getTableFilename(self, seqNum):
        """Return the table filename, or None if not found for the dataId"""
        expId = self._expIdFromSeqNum(seqNum)
        try:
            tablePath = self.butler.getUri('spectractorOutputRoot', expId=expId)
        except butlerExceptions.NoResults:
            return None

        tableFilename = os.path.join(tablePath, 'extractedLines.fits')
        return tableFilename

    def _readOneExtractionFile(self, seqNum):
        filename = self._getTableFilename(seqNum)  # whole thing should really just be a butler.get()
        if not filename:
            return None

        if os.path.exists(filename):
            table = Table.read(filename)
            return table
        return None

    def _loadExtractions(self):
        allSeqNums = self.butler.queryMetadata('raw', 'seqNum', dayObs=self.dayObs, object=self.targetName)

        self.seqNums = []
        for seqNum in allSeqNums:
            if self.isDispersed(seqNum):
                self.seqNums.append(seqNum)
        print(f"Found {len(self.seqNums)} dispersed observations of "
              f"{self.targetName} on {self.dayObs} in registry")
        self.seqNums = sorted(self.seqNums)  # not guaranteed to be in order, I don't think

        self.data = {}
        for seqNum in self.seqNums:
            linesTable = self._readOneExtractionFile(seqNum)
            if linesTable:
                self.data[seqNum] = linesTable
        print(f"Loaded extractions for {len(self.data.keys())} of the above")
        return

    def getFilterDisperserSet(self):
        fullFilters = set()
        for seqNum in self.seqNums:
            fullFilters.add(self.butler.queryMetadata('raw', 'filter', dayObs=self.dayObs, seqNum=seqNum)[0])
        return fullFilters

    def getFilterSet(self):
        filters = set()
        for seqNum in self.seqNums:
            filt = self.butler.queryMetadata('raw', 'filter', dayObs=self.dayObs, seqNum=seqNum)[0]
            filt = filt.split(FILTER_DELIMITER)[0]
            filters.add(filt)
        return filters

    def getDisperserSet(self):
        dispersers = set()
        for seqNum in self.seqNums:
            filt = self.butler.queryMetadata('raw', 'filter', dayObs=self.dayObs, seqNum=seqNum)[0]
            disperser = filt.split(FILTER_DELIMITER)[1]
            dispersers.add(disperser)
        return dispersers

    def getLineValue(self, seqNum, lineName, columnName):
        # just convenient to be able to call this on a class instance as well
        # as the free floating function - nothing deep happening here
        return getLineValue(self.data[seqNum], lineName, columnName)

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
        dates = [self.butler.get('raw_md', dayObs=self.dayObs, seqNum=seqNum)['DATE-OBS']
                 for seqNum in self.seqNums]
        return [Time(d).mjd for d in dates]

    def getAirmasses(self):
        return [airMassFromRawMetadata(self.butler.get('raw_md', dayObs=self.dayObs, seqNum=seqNum))
                for seqNum in self.seqNums]

    def printObservationTable(self):
        for seqNum in self.seqNums:
            filt = self.butler.queryMetadata('raw', 'filter', dayObs=self.dayObs, seqNum=seqNum)[0]
            print(seqNum, filt)
