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

import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase

from lsst.meas.algorithms import LoadIndexedReferenceObjectsTask, MagnitudeLimit, ReferenceObjectLoader
from lsst.meas.astrom import AstrometryTask, FitAffineWcsTask
from lsst.pipe.tasks.quickFrameMeasurement import (QuickFrameMeasurementTask)
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
    astromRefObjLoader = pexConfig.ConfigurableField(
        target=LoadIndexedReferenceObjectsTask,
        doc="Reference object loader for astrometric calibration",
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
        # this is a null option now in Gen3 - do not set it here
        # self.astromRefObjLoader.ref_dataset_name

        self.astromRefObjLoader.pixelMargin = 1000

        self.astrometry.wcsFitter.retarget(FitAffineWcsTask)
        self.astrometry.referenceSelector.doMagLimit = True
        self.astrometry.referenceSelector.magLimit.fluxField = "phot_g_mean_flux"
        self.astrometry.matcher.maxRotationDeg = 5.99
        self.astrometry.matcher.maxOffsetPix = 3000
        self.astrometry.sourceSelector['matcher'].minSnr = 10

        magLimit = MagnitudeLimit()
        magLimit.minimum = 1
        magLimit.maximum = 15
        self.astrometry.referenceSelector.magLimit = magLimit


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
                                             config=self.config.astromRefObjLoader, log=self.log)

        # self.makeSubtask('astromRefObjLoader')
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
        originalFilterLabel = inputExp.getFilterLabel()
        inputExp.setFilterLabel(referenceFilterLabel)

        successfulFit = False
        try:
            astromResult = self.astrometry.run(sourceCat=inputSources, exposure=inputExp)
            scatter = astromResult.scatterOnSky.asArcseconds()
            inputExp.setFilterLabel(originalFilterLabel)
            if scatter < 1:
                successfulFit = True

        except Exception:  # TODO: change this semi-naked except
            self.log.warn("Solver failed to run completely")
            inputExp.setFilterLabel(originalFilterLabel)

        if successfulFit:
            target = inputExp.getMetadata()['OBJECT']
            centroid = getTargetCentroidFromWcs(inputExp, target, logger=self.log)
        else:
            result = self.qfmTask.run(inputExp)
            centroid = result.brightestObjCentroid

        result = pipeBase.Struct(atmospecCentroid={'centroid': centroid,
                                                   'astrometricMatch': successfulFit})
        return result
