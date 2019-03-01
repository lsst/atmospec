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

__all__ = ['ProcessStarTask']

import numpy as np
import matplotlib.pyplot as plt

import lsstDebug
import lsst.afw.image as afwImage
from lsst.ip.isr import IsrTask
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.detection as afwDetect
import lsst.afw.geom as afwGeom

from .dispersion import DispersionRelation
from .extraction import SpectralExtractionTask


class ProcessStarTaskConfig(pexConfig.Config):
    """Configuration parameters for ProcessStarTask."""

    isr = pexConfig.ConfigurableField(
        target=IsrTask,
        doc="Task to perform instrumental signature removal",
    )
    extraction = pexConfig.ConfigurableField(
        target=SpectralExtractionTask,
        doc="Task to perform spectral extractions",
    )
    doWrite = pexConfig.Field(
        dtype=bool,
        doc="Write out the results?",
        default=True,
    )
    mainSourceFindingMethod = pexConfig.ChoiceField(
        doc="Which attribute to prioritise when selecting the main source object",
        dtype=str,
        default="BRIGHTEST",
        allowed={
            "BRIGHTEST": "Select the brightest object with roundness > roundnessCut",
            "ROUNDEST": "Select the roundest object with brightness > fluxCut",
        }
    )
    mainStarRoundnessCut = pexConfig.Field(
        dtype=float,
        doc="Value of ellipticity above which to reject the brightest object."
        " Ignored if mainSourceFindingMethod == BRIGHTEST",
        default=0.2
    )
    mainStarFluxCut = pexConfig.Field(
        dtype=float,
        doc="Object flux below which to reject the roundest object."
        " Ignored if mainSourceFindingMethod == ROUNDEST",
        default=1e7
    )
    mainStarNpixMin = pexConfig.Field(
        dtype=int,
        doc="Minimum number of pixels for object detection of main star",
        default=10
    )
    mainStarNsigma = pexConfig.Field(
        dtype=int,
        doc="nSigma for detection of main star",
        default=200  # the m=0 is very bright indeed, and we don't want to detect much spectrum
    )
    mainStarGrow = pexConfig.Field(
        dtype=int,
        doc="Number of pixels to grow by when detecting main star. This"
        " encourages the spectrum to merge into one footprint, but too much"
        " makes everything round, compromising mainStarRoundnessCut's"
        " effectiveness",
        default=5
    )
    mainStarGrowIsotropic = pexConfig.Field(
        dtype=bool,
        doc="Grow main star's footprint isotropically?",
        default=False
    )
    aperture = pexConfig.Field(
        dtype=int,
        doc="Width of the aperture to use in pixels",
        default=250
    )
    spectrumLengthPixels = pexConfig.Field(
        dtype=int,
        doc="Length of the spectrum in pixels",
        default=5000
    )
    offsetFromMainStar = pexConfig.Field(
        dtype=int,
        doc="Number of pixels from the main star's centroid to start extraction",
        default=100
    )
    dispersionDirection = pexConfig.ChoiceField(
        doc="Direction along which the image is dispersed",
        dtype=str,
        default="y",
        allowed={
            "x": "Dispersion along the serial direction",
            "y": "Dispersion along the parallel direction",
        }
    )
    spectralOrder = pexConfig.ChoiceField(
        doc="Direction along which the image is dispersed",
        dtype=str,
        default="+1",
        allowed={
            "+1": "Use the m+1 spectrum",
            "-1": "Use the m-1 spectrum",
            "both": "Use both spectra",
        }
    )
    doFlat = pexConfig.Field(
        dtype=bool,
        doc="Flatfield the image?",
        default=True
    )
    doCosmics = pexConfig.Field(
        dtype=bool,
        doc="Repair cosmic rays?",
        default=True
    )


class ProcessStarTask(pipeBase.CmdLineTask):
    """Task for the spectral extraction of single-star dispersed images.

    For a full description of how this tasks works, see the run() method.
    """

    ConfigClass = ProcessStarTaskConfig
    _DefaultName = "processStar"

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("extraction")

        self.debug = lsstDebug.Info(__name__)
        if self.debug.enabled:
            self.log.info("Running with debug enabled...")
            # If we're displaying, test it works and save displays for later.
            # It's worth testing here as displays are flaky and sometimes
            # can't be contacted, and given processing takes a while,
            # it's a shame to fail late due to display issues.
            if self.debug.display:
                try:
                    import lsst.afw.display as afwDisp
                    afwDisp.setDefaultBackend(self.debug.displayBackend)
                    afwDisp.Display.delAllDisplays()
                    # pick an unlikely number to be safe xxx replace this
                    self.disp1 = afwDisp.Display(987, open=True)

                    im = afwImage.ImageF(2, 2)
                    im.array[:] = np.ones((2, 2))
                    self.disp1.mtv(im)
                    self.disp1.erase()
                    afwDisp.setDefaultMaskTransparency(90)
                except NameError:
                    self.debug.display = False
                    self.log.warn('Failed to setup/connect to display! Debug display has been disabled')

        if self.debug.notheadless:
            pass  # other backend options can go here
        else:  # this stop windows popping up when plotting. When headless, use 'agg' backend too
            plt.interactive(False)

        self.config.validate()
        self.config.freeze()

    def findObjects(self, exp, nSigma=None, grow=0):
        """Find the objects in a postISR exposure."""
        nPixMin = self.config.mainStarNpixMin
        if not nSigma:
            nSigma = self.config.mainStarNsigma
        if not grow:
            grow = self.config.mainStarGrow
            isotropic = self.config.mainStarGrowIsotropic

        threshold = afwDetect.Threshold(nSigma, afwDetect.Threshold.STDEV)
        footPrintSet = afwDetect.FootprintSet(exp.getMaskedImage(), threshold, "DETECTED", nPixMin)
        if grow > 0:
            footPrintSet = afwDetect.FootprintSet(footPrintSet, grow, isotropic)
        return footPrintSet

    def _getEllipticity(self, shape):
        """Calculate the ellipticity given a quadrupole shape.

        Parameters
        ----------
        shape : `lsst.afw.geom.ellipses.Quadrupole`
            The quadrupole shape

        Returns
        -------
        ellipticity : `float`
            The magnitude of the ellipticity
        """
        ixx = shape.getIxx()
        iyy = shape.getIyy()
        ixy = shape.getIxy()
        ePlus = (ixx - iyy) / (ixx + iyy)
        eCross = 2*ixy / (ixx + iyy)
        return (ePlus**2 + eCross**2)**0.5

    def getRoundestObject(self, footPrintSet, parentExp, fluxCut=1e-15):
        """Get the roundest object brighter than fluxCut from a footPrintSet.

        Parameters
        ----------
        footPrintSet : `lsst.afw.detection.FootprintSet`
            The set of footprints resulting from running detection on parentExp

        parentExp : `lsst.afw.image.exposure`
            The parent exposure for the footprint set.

        fluxCut : `float`
            The flux, below which, sources are rejected.

        Returns
        -------
        source : `lsst.afw.detection.Footprint`
            The winning footprint from the input footPrintSet
        """
        self.log.debug("ellipticity\tflux/1e6\tcentroid")
        sourceDict = {}
        for fp in footPrintSet.getFootprints():
            shape = fp.getShape()
            e = self._getEllipticity(shape)
            flux = fp.getSpans().flatten(parentExp.image.array, parentExp.image.getXY0()).sum()
            self.log.debug("%.4f\t%.2f\t%s"%(e, flux/1e6, str(fp.getCentroid())))
            if flux > fluxCut:
                sourceDict[e] = fp

        return sourceDict[sorted(sourceDict.keys())[0]]

    def getBrightestObject(self, footPrintSet, parentExp, roundnessCut=1e9):
        """Get the brightest object rounder than the cut from a footPrintSet.

        Parameters
        ----------
        footPrintSet : `lsst.afw.detection.FootprintSet`
            The set of footprints resulting from running detection on parentExp

        parentExp : `lsst.afw.image.exposure`
            The parent exposure for the footprint set.

        roundnessCut : `float`
            The ellipticity, above which, sources are rejected.

        Returns
        -------
        source : `lsst.afw.detection.Footprint`
            The winning footprint from the input footPrintSet
        """
        self.log.debug("ellipticity\tflux\tcentroid")
        sourceDict = {}
        for fp in footPrintSet.getFootprints():
            shape = fp.getShape()
            e = self._getEllipticity(shape)
            flux = fp.getSpans().flatten(parentExp.image.array, parentExp.image.getXY0()).sum()
            self.log.debug("%.4f\t%.2f\t%s"%(e, flux/1e6, str(fp.getCentroid())))
            if e < roundnessCut:
                sourceDict[flux] = fp

        return sourceDict[sorted(sourceDict.keys())[-1]]

    def findMainSource(self, exp):
        """Return the x,y of the brightest or roundest object in an exposure.

        Given a postISR exposure, run source detection on it, and return the
        centroid of the main star. Depending on the task configuration, this
        will either be the roundest object above a certain flux cutoff, or
        the brightest object which is rounder than some ellipticity cutoff.

        Parameters
        ----------
        exp : `afw.image.Exposure`
            The postISR exposure in which to find the main star

        Returns
        -------
        x, y : `tuple` of `float`
            The centroid of the main star in the image

        Notes
        -----
        Behaviour of this method is controlled by many task config params
        including, for the detection stage:
        config.mainStarNpixMin
        config.mainStarNsigma
        config.mainStarGrow
        config.mainStarGrowIsotropic

        And post-detection, for selecting the main source:
        config.mainSourceFindingMethod
        config.mainStarFluxCut
        config.mainStarRoundnessCut
        """
        fpSet = self.findObjects(exp)
        if self.config.mainSourceFindingMethod == 'ROUNDEST':
            source = self.getRoundestObject(fpSet, exp, fluxCut=self.config.mainStarFluxCut)
        elif self.config.mainSourceFindingMethod == 'BRIGHTEST':
            source = self.getBrightestObject(fpSet, exp,
                                             roundnessCut=self.config.mainStarRoundnessCut)
        else:
            # should be impossible as this is a choice field, but still
            raise RuntimeError(f"Invalid source finding method"
                               "selected: {self.mainSourceFindingMethod}")
        return source.getCentroid()

    def runDataRef(self, dataRef):
        """Run the ProcessStarTask on a ButlerDataRef for a single visit.

        Runs isr to get the postISR exposure from the dataRef and passes this
        to the run() method.

        Parameters
        ----------
        dataRef : `daf.persistence.butlerSubset.ButlerDataRef`
            Butler reference of the detector and visit
        """
        self.log.info("Processing %s" % (dataRef.dataId))
        exposure = self.isr.runDataRef(dataRef).exposure
        self.run(exposure)

        return

    def run(self, exp):
        """Calculate the wavelength calibrated 1D spectrum from a postISRCCD.

        An outline of the steps in the processing is as follows:
         * Source extraction - find the objects in image
         * Process sources to find the x,y of the main star

         * Given the centroid, the dispersion direction, and the order(s),
           calculate the spectrum's bounding box

         * (Rotate the image such that the dispersion direction is vertical
            TODO: DM-18138)

         * Create an initial dispersion relation object from the geometry
           or alternative bootstrapping method

         * Apply an initial flatfielding - TODO: DM-18141

         * Find and interpolate over cosmics if necessary - TODO: DM-18140

         * Perform an initial spectral extraction, depending on selected method
         *     Fit a background model and subtract
         *     Perform row-wise fits for extraction
         *     TODO: DM-18136 for doing a full-spectrum fit with PSF model

         * Given knowledge of features in the spectrum, find lines in the
           measured spectrum and re-fit to refine the dispersion relation
         * Reflatfield the image with the refined dispersion relation

        Parameters
        ----------
        exp : `afw.image.Exposure`
            The postISR exposure in which to find the main star

        Returns
        -------
        spectrum : `lsst.atmospec.spectrum` - TODO: DM-18133
            The wavelength-calibrated 1D stellar spectrum
        """

        sourceCentroid = self.findMainSource(exp)
        self.log.info("Centroid of main star at: {}".format(sourceCentroid))

        spectrumBbox = self.calcSpectrumBbox(exp, sourceCentroid, self.config.aperture,
                                             self.config.spectralOrder)
        self.log.info("Spectrum bbox = {}".format(spectrumBbox))

        if self.debug.display and 'raw' in self.debug.displayItems:
            self.disp1.mtv(exp)
            self.log.info("Showing full postISR image")
            self.log.info("Centroid of main star at: {}".format(sourceCentroid))
            self.log.info("Spectrum bbox will be at: {}".format(spectrumBbox))
            input("Press return to continue...")
        if self.debug.display and 'spectrum' in self.debug.displayItems:
            self.log.info("Showing spectrum image using bbox {}".format(spectrumBbox))
            self.disp1.mtv(exp[spectrumBbox])

        disp = DispersionRelation(None, [2, 4, 6])

        if self.config.doFlat:
            exp = self.flatfield(exp, disp)

        if self.config.doCosmics:
            exp = self.repairCosmics(exp, disp)

        self.returnForNotebook = [exp, spectrumBbox]  # xxx remove this, just for notebook playing

        spectrum = self.measureSpectrum(exp, sourceCentroid, spectrumBbox, disp)

        self.returnForNotebook.append(spectrum)

        return

    def flatfield(self, exp, disp):
        """Placeholder for wavelength dependent flatfielding. See TODO: DM-18141

        Will probably need a dataRef, as it will need to be retrieving flats
        over a range. Also, it will be somewhat complex, so probably needs
        moving to its own task"""
        self.log.warn("Flatfielding not yet implemented")
        return exp

    def repairCosmics(self, exp, disp):
        self.log.warn("Cosmic ray repair not yet implemented")
        return exp

    def measureSpectrum(self, exp, sourceCentroid, spectrumBbox, dispersionRelation):
        """Perform the spectral extraction, given a source location and exp."""

        self.extraction.initialise(exp, sourceCentroid, spectrumBbox, dispersionRelation)

        # xxx this method currently doesn't return an object - fix this
        spectrum = self.extraction.getFluxBasic()

        return spectrum

    def calcSpectrumBbox(self, exp, centroid, aperture, order='+1'):
        """Calculate the bbox for the spectrum, given the centroid.

        XXX Longer explanation here, inc. parameters
        TODO: Add support for order = "both"
        """
        extent = self.config.spectrumLengthPixels
        halfWidth = aperture//2
        translate_x = self.config.offsetFromMainStar
        translate_y = self.config.offsetFromMainStar
        sourceX = centroid[0]
        sourceY = centroid[1]

        if(order == '-1'):
            translate_x = - extent - self.config.offsetFromMainStar
            translate_y = - extent - self.config.offsetFromMainStar

        if (self.config.dispersionDirection == 'x'):
            xStart = sourceX + translate_x
            xEnd = xStart + extent - 1
            yStart = sourceY - halfWidth
            yEnd = sourceY + halfWidth - 1
        elif(self.config.dispersionDirection == 'y'):
            xStart = sourceX - halfWidth
            xEnd = sourceX + halfWidth - 1
            yStart = sourceY + translate_y
            yEnd = yStart + extent - 1

        xEnd = min(xEnd, exp.getWidth())
        yEnd = min(yEnd, exp.getHeight()-1)
        yStart = max(yStart, 0)
        xStart = max(xStart, 0)
        assert (xEnd > xStart) and (yEnd > yStart)

        self.log.debug('(xStart, xEnd) = (%s, %s)'%(xStart, xEnd))
        self.log.debug('(yStart, yEnd) = (%s, %s)'%(yStart, yEnd))

        bbox = afwGeom.Box2I(afwGeom.Point2I(xStart, yStart), afwGeom.Point2I(xEnd, yEnd))
        return bbox
