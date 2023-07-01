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

import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase

from lsst.meas.algorithms import LoadReferenceObjectsConfig, MagnitudeLimit, ReferenceObjectLoader
from lsst.meas.astrom import AstrometryTask, FitAffineWcsTask
from lsst.pipe.tasks.quickFrameMeasurement import (QuickFrameMeasurementTask)
from lsst.pipe.base.task import TaskError
import lsst.pipe.base.connectionTypes as cT
import lsst.pex.config as pexConfig

from .utils import getTargetCentroidFromWcs

__all__ = ['SingleStarCentroidTaskConfig', 'SingleStarCentroidTask']


class SingleStarCentroidTaskConnections(pipeBase.PipelineTaskConnections,
                                        dimensions=("instrument", "visit", "detector")):
    inputExp = cT.Input(
        name="icExp",
        doc="Image-characterize output exposure",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
        multiple=False,
    )
    inputSources = cT.Input(
        name="icSrc",
        doc="Image-characterize output sources.",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector"),
        multiple=False,
    )
    astromRefCat = cT.PrerequisiteInput(
        doc="Reference catalog to use for astrometry",
        name="gaia_dr2_20200414",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        deferLoad=True,
        multiple=True,
    )
    atmospecCentroid = cT.Output(
        name="atmospecCentroid",
        doc="The main star centroid in yaml format.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "visit", "detector"),
    )


class SingleStarCentroidTaskConfig(pipeBase.PipelineTaskConfig,
                                   pipelineConnections=SingleStarCentroidTaskConnections):
    """Configuration parameters for ProcessStarTask."""
    astromRefObjLoader = pexConfig.ConfigField(
        dtype=LoadReferenceObjectsConfig,
        doc="Reference object loader config for astrometric calibration.",
    )
    astrometry = pexConfig.ConfigurableField(
        target=AstrometryTask,
        doc="Task to perform astrometric calibration to refine the WCS",
    )
    qfmTask = pexConfig.ConfigurableField(
        target=QuickFrameMeasurementTask,
        doc="XXX",
    )
    referenceFilterOverride = pexConfig.Field(
        dtype=str,
        doc="Which filter in the reference catalog to match to?",
        default="phot_g_mean"
    )

    def setDefaults(self):
        super().setDefaults()
        self.astromRefObjLoader.pixelMargin = 1000

        self.astrometry.wcsFitter.retarget(FitAffineWcsTask)
        self.astrometry.referenceSelector.doMagLimit = True
        magLimit = MagnitudeLimit()
        magLimit.minimum = 1
        magLimit.maximum = 15
        self.astrometry.referenceSelector.magLimit = magLimit
        self.astrometry.referenceSelector.magLimit.fluxField = "phot_g_mean_flux"
        self.astrometry.matcher.maxRotationDeg = 5.99
        self.astrometry.matcher.maxOffsetPix = 3000
        self.astrometry.sourceSelector['matcher'].minSnr = 10


class SingleStarCentroidTask(pipeBase.PipelineTask):
    """XXX Docs here
    """

    ConfigClass = SingleStarCentroidTaskConfig
    _DefaultName = 'singleStarCentroid'

    def __init__(self, initInputs=None, **kwargs):
        super().__init__(**kwargs)

        self.makeSubtask("astrometry", refObjLoader=None)
        self.makeSubtask('qfmTask')

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        refObjLoader = ReferenceObjectLoader(dataIds=[ref.datasetRef.dataId
                                                      for ref in inputRefs.astromRefCat],
                                             refCats=inputs.pop('astromRefCat'),
                                             name=self.config.connections.astromRefCat,
                                             config=self.config.astromRefObjLoader, log=self.log)

        refObjLoader.pixelMargin = 1000
        self.astrometry.setRefObjLoader(refObjLoader)

        # See L603 (def runQuantum(self, butlerQC, inputRefs, outputRefs):)
        # in calibrate.py to put photocal back in

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputExp, inputSources):
        """XXX Docs
        """

        # TODO: Change this to doing this the proper way
        referenceFilterName = self.config.referenceFilterOverride
        referenceFilterLabel = afwImage.FilterLabel(physical=referenceFilterName, band=referenceFilterName)
        # there's a better way of doing this with the task I think
        originalFilterLabel = inputExp.getFilter()
        inputExp.setFilter(referenceFilterLabel)
        originalWcs = inputExp.getWcs()

        successfulFit = False
        try:
            astromResult = self.astrometry.run(sourceCat=inputSources, exposure=inputExp)
            scatter = astromResult.scatterOnSky.asArcseconds()
            inputExp.setFilter(originalFilterLabel)
            if scatter < 1:
                successfulFit = True
        except (RuntimeError, TaskError, IndexError, ValueError, AttributeError) as e:
            # IndexError raised for low source counts:
            # index 0 is out of bounds for axis 0 with size 0

            # ValueError: negative dimensions are not allowed
            # seen when refcat source count is low (but non-zero)

            # AttributeError: 'NoneType' object has no attribute 'asArcseconds'
            # when the result is a failure as the wcs is set to None on failure
            self.log.warn(f"Solving failed: {e}")
            inputExp.setWcs(originalWcs)  # this is set to None when the fit fails, so restore it
        finally:
            inputExp.setFilter(originalFilterLabel)  # always restore this

        centroid = None
        if successfulFit:
            target = inputExp.visitInfo.object
            centroid = getTargetCentroidFromWcs(inputExp, target, logger=self.log)
            if not centroid:
                successfulFit = False
                self.log.warning(f'Failed to find target centroid for {target} via WCS')
        if not centroid:
            result = self.qfmTask.run(inputExp)
            centroid = result.brightestObjCentroid

        centroidTuple = (centroid[0], centroid[1])  # unify Point2D or tuple to tuple
        self.log.info(f"Centroid of main star found at {centroidTuple} found"
                      f" via {'astrometry' if successfulFit else 'QuickFrameMeasurement'}")
        result = pipeBase.Struct(atmospecCentroid={'centroid': centroidTuple,
                                                   'astrometricMatch': successfulFit})
        return result
