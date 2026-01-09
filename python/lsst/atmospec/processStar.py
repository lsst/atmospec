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

__all__ = ['ProcessStarTask', 'ProcessStarTaskConfig']

import importlib.resources
import shutil
import numpy as np
import matplotlib.pyplot as plt

import lsstDebug
import lsst.afw.image as afwImage
import lsst.geom as geom
from lsst.ip.isr import IsrTaskLSST
import lsst.pex.config as pexConfig
from lsst.pex.config import FieldValidationError
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pipe.base.task import TaskError

from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask
from lsst.meas.algorithms import ReferenceObjectLoader
from lsst.meas.astrom import AstrometryTask, FitAffineWcsTask

import lsst.afw.detection as afwDetect

from .spectraction import SpectractorShim
from .utils import getLinearStagePosition, isDispersedExp, getFilterAndDisperserFromExp

COMMISSIONING = False  # allows illegal things for on the mountain usage.

# TODO:
# Sort out read noise and gain
# remove dummy image totally
# talk to Jeremy about turning the image beforehand and giving new coords
# deal with not having ambient temp
# Gen3ification
# astropy warning for units on save
# but actually just remove all manual saves entirely, I think?
# Make SED persistable
# Move to QFM for star finding failover case
# Remove old cruft functions
# change spectractions run method to be ~all kwargs with *,...


class ProcessStarTaskConnections(pipeBase.PipelineTaskConnections,
                                 dimensions=("instrument", "visit", "detector")):
    inputExp = cT.Input(
        name="icExp",
        doc="Image-characterize output exposure",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
        multiple=False,
    )
    inputCentroid = cT.Input(
        name="atmospecCentroid",
        doc="The main star centroid in yaml format.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "visit", "detector"),
        multiple=False,
    )
    spectractorSpectrum = cT.Output(
        name="spectractorSpectrum",
        doc="The Spectractor output spectrum.",
        storageClass="SpectractorSpectrum",
        dimensions=("instrument", "visit", "detector"),
    )
    spectractorImage = cT.Output(
        name="spectractorImage",
        doc="The Spectractor output image.",
        storageClass="SpectractorImage",
        dimensions=("instrument", "visit", "detector"),
    )
    spectrumForwardModelFitParameters = cT.Output(
        name="spectrumForwardModelFitParameters",
        doc="The full forward model fit parameters.",
        storageClass="SpectractorFitParameters",
        dimensions=("instrument", "visit", "detector"),
    )
    spectrumLibradtranFitParameters = cT.Output(
        name="spectrumLibradtranFitParameters",
        doc="The fitted Spectractor atmospheric parameters from fitting the atmosphere with libradtran"
            " on the spectrum.",
        storageClass="SpectractorFitParameters",
        dimensions=("instrument", "visit", "detector"),
    )
    spectrogramLibradtranFitParameters = cT.Output(
        name="spectrogramLibradtranFitParameters",
        doc="The fitted Spectractor atmospheric parameters from fitting the atmosphere with libradtran"
            " directly on the spectrogram.",
        storageClass="SpectractorFitParameters",
        dimensions=("instrument", "visit", "detector"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.doFullForwardModelDeconvolution:
            self.outputs.remove("spectrumForwardModelFitParameters")
        if not config.doFitAtmosphere:
            self.outputs.remove("spectrumLibradtranFitParameters")
        if not config.doFitAtmosphereOnSpectrogram:
            self.outputs.remove("spectrogramLibradtranFitParameters")


class ProcessStarTaskConfig(pipeBase.PipelineTaskConfig,
                            pipelineConnections=ProcessStarTaskConnections):
    """Configuration parameters for ProcessStarTask."""
    # Spectractor parameters:
    targetCentroidMethod = pexConfig.ChoiceField(
        dtype=str,
        doc="Method to get target centroid. "
        "SPECTRACTOR_FIT_TARGET_CENTROID internally.",
        default="auto",
        allowed={
            # note that although this config option controls
            # SPECTRACTOR_FIT_TARGET_CENTROID, it doesn't map there directly,
            # because Spectractor only has the concepts of guess, fit and wcs,
            # and it calls "exact" "guess" internally, so that's remapped.
            "auto": "If the upstream astrometric fit succeeded, and therefore"
            " the centroid is an exact one, use that as an ``exact`` value,"
            " otherwise tell Spectractor to ``fit`` the centroid",
            "exact": "Use a given input value as source of truth.",
            "fit": "Fit a 2d Moffat model to the target.",
            "WCS": "Use the target's catalog location and the image's wcs.",
        }
    )
    rotationAngleMethod = pexConfig.ChoiceField(
        dtype=str,
        doc="Method used to get the image rotation angle. "
        "SPECTRACTOR_COMPUTE_ROTATION_ANGLE internally.",
        default="disperser",
        allowed={
            # XXX MFL: probably need to use setDefaults to set this based on
            # the disperser. I think Ronchi gratings want hessian and the
            # holograms want disperser.
            "False": "Do not rotate the image.",
            "disperser": "Use the disperser angle geometry as specified in the disperser definition.",
            "hessian": "Compute the angle from the image using a Hessian transform.",
        }
    )
    doDeconvolveSpectrum = pexConfig.Field(
        dtype=bool,
        doc="Deconvolve the spectrogram with a simple 2D PSF analysis? "
        "SPECTRACTOR_DECONVOLUTION_PSF2D internally.",
        default=True,
    )
    doFullForwardModelDeconvolution = pexConfig.Field(
        dtype=bool,
        doc="Deconvolve the spectrogram with full forward model? "
        "SPECTRACTOR_DECONVOLUTION_FFM internally.",
        default=True,
    )
    deconvolutionSigmaClip = pexConfig.Field(
        dtype=float,
        doc="Sigma clipping level for the deconvolution when fitting the full forward model? "
        "SPECTRACTOR_DECONVOLUTION_SIGMA_CLIP internally.",
        default=100,
    )
    doSubtractBackground = pexConfig.Field(
        dtype=bool,
        doc="Subtract the background with Spectractor? "
        "SPECTRACTOR_BACKGROUND_SUBTRACTION internally.",
        default=True,
    )
    rebin = pexConfig.Field(
        dtype=int,
        doc="Rebinning factor to use on the input image, in pixels. "
        "CCD_REBIN internally.",
        default=2,  # TODO Change to 1 once speed issues are resolved
    )
    xWindow = pexConfig.Field(
        dtype=int,
        doc="Window x size to search for the target object. Ignored if targetCentroidMethod in ('exact, wcs')"
        "XWINDOW internally.",
        default=150,
    )
    yWindow = pexConfig.Field(
        dtype=int,
        doc="Window y size to search for the targeted object. Ignored if targetCentroidMethod in "
        "('exact, wcs')"
        "YWINDOW internally.",
        default=150,
    )
    xWindowRotated = pexConfig.Field(
        dtype=int,
        doc="Window x size to search for the target object in the rotated image. "
        "Ignored if rotationAngleMethod=False"
        "XWINDOW_ROT internally.",
        default=50,
    )
    yWindowRotated = pexConfig.Field(
        dtype=int,
        doc="Window y size to search for the target object in the rotated image. "
        "Ignored if rotationAngleMethod=False"
        "YWINDOW_ROT internally.",
        default=50,
    )
    pixelShiftPrior = pexConfig.Field(
        dtype=float,
        doc="Prior on the reliability of the centroid estimate in pixels. "
        "PIXSHIFT_PRIOR internally.",
        default=5,
        check=lambda x: x > 0,
    )
    doFilterRotatedImage = pexConfig.Field(
        dtype=bool,
        doc="Apply a filter to the rotated image? If not True, this creates residuals and correlated noise. "
        "ROT_PREFILTER internally.",
        default=True,
    )
    imageRotationSplineOrder = pexConfig.Field(
        dtype=int,
        doc="Order of the spline used when rotating the image. "
        "ROT_ORDER internally.",
        default=5,
        # XXX min value of 3 for allowed range, max 5
    )
    rotationAngleMin = pexConfig.Field(
        dtype=float,
        doc="In the Hessian analysis to compute the rotation angle, cut all angles below this, in degrees. "
        "ROT_ANGLE_MIN internally.",
        default=-10,
    )
    rotationAngleMax = pexConfig.Field(
        dtype=float,
        doc="In the Hessian analysis to compute rotation angle, cut all angles above this, in degrees. "
        "ROT_ANGLE_MAX internally.",
        default=10,
    )
    plotLineWidth = pexConfig.Field(
        dtype=float,
        doc="Line width parameter for plotting. "
        "LINEWIDTH internally.",
        default=2,
    )
    verbose = pexConfig.Field(
        dtype=bool,
        doc="Set verbose mode? "
        "VERBOSE internally.",
        default=True,  # sets INFO level logging in Spectractor
    )
    spectractorDebugMode = pexConfig.Field(
        dtype=bool,
        doc="Set spectractor debug mode? "
        "DEBUG internally.",
        default=True,
    )
    spectractorDebugLogging = pexConfig.Field(
        dtype=bool,
        doc="Set spectractor debug logging? "
        "DEBUG_LOGGING internally.",
        default=False
    )
    doDisplay = pexConfig.Field(
        dtype=bool,
        doc="Display plots, for example when running in a notebook? "
        "DISPLAY internally.",
        default=True
    )
    lambdaMin = pexConfig.Field(
        dtype=int,
        doc="Minimum wavelength for spectral extraction (in nm). "
        "LAMBDA_MIN internally.",
        default=300
    )
    lambdaMax = pexConfig.Field(
        dtype=int,
        doc=" maximum wavelength for spectrum extraction (in nm). "
        "LAMBDA_MAX internally.",
        default=1100
    )
    lambdaStep = pexConfig.Field(
        dtype=float,
        doc="Step size for the wavelength array (in nm). "
        "LAMBDA_STEP internally.",
        default=1,
    )
    spectralOrder = pexConfig.ChoiceField(
        dtype=int,
        doc="The spectral order to extract. "
        "SPEC_ORDER internally.",
        default=1,
        allowed={
            1: "The first order spectrum in the positive y direction",
            -1: "The first order spectrum in the negative y direction",
            2: "The second order spectrum in the positive y direction",
            -2: "The second order spectrum in the negative y direction",
        }
    )
    signalWidth = pexConfig.Field(  # TODO: change this to be set wrt the focus/seeing, i.e. FWHM from imChar
        dtype=int,
        doc="Half transverse width of the signal rectangular window in pixels. "
        "PIXWIDTH_SIGNAL internally.",
        default=40,
    )
    backgroundDistance = pexConfig.Field(
        dtype=int,
        doc="Distance from dispersion axis to analyse the background in pixels. "
        "PIXDIST_BACKGROUND internally.",
        default=140,
    )
    backgroundWidth = pexConfig.Field(
        dtype=int,
        doc="Transverse width of the background rectangular window in pixels. "
        "PIXWIDTH_BACKGROUND internally.",
        default=40,
    )
    backgroundBoxSize = pexConfig.Field(
        dtype=int,
        doc="Box size for sextractor evaluation of the background. "
        "PIXWIDTH_BOXSIZE internally.",
        default=20,
    )
    backgroundOrder = pexConfig.Field(
        dtype=int,
        doc="The order of the polynomial background to fit in the transverse direction. "
        "BGD_ORDER internally.",
        default=1,
    )
    psfType = pexConfig.ChoiceField(
        dtype=str,
        doc="The PSF model type to use. "
        "PSF_TYPE internally.",
        default="Moffat",
        allowed={
            "Moffat": "A Moffat function",
            "MoffatGauss": "A Moffat plus a Gaussian"
        }
    )
    psfPolynomialOrder = pexConfig.Field(
        dtype=int,
        doc="The order of the polynomials to model wavelength dependence of the PSF shape parameters. "
        "PSF_POLY_ORDER internally.",
        default=2
    )
    psfRegularization = pexConfig.Field(
        dtype=float,
        doc="Regularisation parameter for the chisq minimisation to extract the spectrum. "
        "PSF_FIT_REG_PARAM internally.",
        default=1,
        # XXX allowed range strictly positive
    )
    psfTransverseStepSize = pexConfig.Field(
        dtype=int,
        doc="Step size in pixels for the first transverse PSF1D fit. "
        "PSF_PIXEL_STEP_TRANSVERSE_FIT internally.",
        default=10,
    )
    psfFwhmClip = pexConfig.Field(
        dtype=float,
        doc="PSF is not evaluated outside a region larger than max(signalWidth, psfFwhmClip*fwhm) pixels. "
        "PSF_FWHM_CLIP internally.",
        default=2,
    )
    calibBackgroundOrder = pexConfig.Field(
        dtype=int,
        doc="Order of the background polynomial to fit. "
        "CALIB_BGD_ORDER internally.",
        default=3,
    )
    calibPeakWidth = pexConfig.Field(
        dtype=int,
        doc="Half-range to look for local extrema in pixels around tabulated line values. "
        "CALIB_PEAK_WIDTH internally.",
        default=7
    )
    calibBackgroundWidth = pexConfig.Field(
        dtype=int,
        doc="Size of the peak sides to use to fit spectrum base line. "
        "CALIB_BGD_WIDTH internally.",
        default=15,
    )
    calibSavgolWindow = pexConfig.Field(
        dtype=int,
        doc="Window size for the savgol filter in pixels. "
        "CALIB_SAVGOL_WINDOW internally.",
        default=5,
    )
    calibSavgolOrder = pexConfig.Field(
        dtype=int,
        doc="Polynomial order for the savgol filter. "
        "CALIB_SAVGOL_ORDER internally.",
        default=2,
    )
    transmissionSystematicError = pexConfig.Field(
        dtype=float,
        doc="The systematic error on the instrumental transmission. OBS_TRANSMISSION_SYSTEMATICS internally",
        default=0.005
    )
    instrumentTransmissionOverride = pexConfig.Field(
        dtype=str,
        doc="File to use for the full instrumental transmission. Must be located in the"
        " $SPECTRACTOR_DIR/spectractor/simulation/AuxTelThroughput/ directory."
        " OBS_FULL_INSTRUMENT_TRANSMISSON internally.",
        default="multispectra_holo4_003_HD142331_AuxTel_throughput.txt"
    )
    offsetFromMainStar = pexConfig.Field(
        dtype=int,
        doc="Number of pixels from the main star's centroid to start extraction",
        default=100
    )
    spectrumLengthPixels = pexConfig.Field(
        dtype=int,
        doc="Length of the spectrum in pixels",
        default=5000
    )
    # ProcessStar own parameters
    isr = pexConfig.ConfigurableField(
        target=IsrTaskLSST,
        doc="Task to perform instrumental signature removal",
    )
    charImage = pexConfig.ConfigurableField(
        target=CharacterizeImageTask,
        doc="""Task to characterize a science exposure:
            - detect sources, usually at high S/N
            - estimate the background, which is subtracted from the image and returned as field "background"
            - estimate a PSF model, which is added to the exposure
            - interpolate over defects and cosmic rays, updating the image, variance and mask planes
            """,
    )
    doWrite = pexConfig.Field(
        dtype=bool,
        doc="Write out the results?",
        default=True,
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
    doDisplayPlots = pexConfig.Field(
        dtype=bool,
        doc="Matplotlib show() the plots, so they show up in a notebook or X window",
        default=False
    )
    doSavePlots = pexConfig.Field(
        dtype=bool,
        doc="Save matplotlib plots to output rerun?",
        default=False
    )
    forceObjectName = pexConfig.Field(
        dtype=str,
        doc="A supplementary name for OBJECT. Will be forced to apply to ALL visits, so this should only"
            " ONLY be used for immediate commissioning debug purposes. All long term fixes should be"
            " supplied as header fix-up yaml files.",
        default=""
    )
    referenceFilterOverride = pexConfig.Field(
        dtype=str,
        doc="Which filter in the reference catalog to match to?",
        default="phot_g_mean"
    )
    # This is a post-processing function in Spectractor and therefore isn't
    # controlled by its top-level function, and thus doesn't map to a
    # spectractor.parameters ALL_CAPS config option
    doFitAtmosphere = pexConfig.Field(
        dtype=bool,
        doc="Use uvspec to fit the atmosphere? Requires the binary to be available.",
        default=False
    )
    # This is a post-processing function in Spectractor and therefore isn't
    # controlled by its top-level function, and thus doesn't map to a
    # spectractor.parameters ALL_CAPS config option
    doFitAtmosphereOnSpectrogram = pexConfig.Field(
        dtype=bool,
        doc="Experimental option to use uvspec to fit the atmosphere directly on the spectrogram?"
            " Requires the binary to be available.",
        default=False
    )

    def setDefaults(self):
        self.charImage.doWriteExposure = False

        self.charImage.doApCorr = False
        self.charImage.doMeasurePsf = False
        self.charImage.repair.cosmicray.nCrPixelMax = 100000
        self.charImage.repair.doCosmicRay = False
        if self.charImage.doMeasurePsf:
            self.charImage.measurePsf.starSelector['objectSize'].signalToNoiseMin = 10.0
            self.charImage.measurePsf.starSelector['objectSize'].fluxMin = 5000.0
        self.charImage.detection.includeThresholdMultiplier = 3

    def validate(self):
        super().validate()
        uvspecPath = shutil.which('uvspec')
        if uvspecPath is None and self.doFitAtmosphere is True:
            raise FieldValidationError(self.__class__.doFitAtmosphere, self, "uvspec is not in the path,"
                                       " but doFitAtmosphere is True.")
        if uvspecPath is None and self.doFitAtmosphereOnSpectrogram is True:
            raise FieldValidationError(self.__class__.doFitAtmosphereOnSpectrogram, self, "uvspec is not in"
                                       " the path, but doFitAtmosphere is True.")


class ProcessStarTask(pipeBase.PipelineTask):
    """Task for the spectral extraction of single-star dispersed images.

    For a full description of how this tasks works, see the run() method.
    """

    ConfigClass = ProcessStarTaskConfig
    _DefaultName = "processStar"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("charImage")

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
                    self.log.warning('Failed to setup/connect to display! Debug display has been disabled')

        if self.debug.notHeadless:
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
        Behavior of this method is controlled by many task config params
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
        # TODO: probably replace all this with QFM
        fpSet = self.findObjects(exp)
        if self.config.mainSourceFindingMethod == 'ROUNDEST':
            source = self.getRoundestObject(fpSet, exp, fluxCut=self.config.mainStarFluxCut)
        elif self.config.mainSourceFindingMethod == 'BRIGHTEST':
            source = self.getBrightestObject(fpSet, exp,
                                             roundnessCut=self.config.mainStarRoundnessCut)
        else:
            # should be impossible as this is a choice field, but still
            raise RuntimeError("Invalid source finding method "
                               f"selected: {self.config.mainSourceFindingMethod}")
        return source.getCentroid()

    def updateMetadata(self, exp, **kwargs):
        """Update an exposure's metadata with set items from the visit info.

        Spectractor expects many items, like the hour angle and airmass, to be
        in the metadata, so pull them out of the visit info etc and put them
        into the main metadata. Also updates the metadata with any supplied
        kwargs.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            The exposure to update.
        **kwargs : `dict`
            The items to add.
        """
        md = exp.getMetadata()
        vi = exp.getInfo().getVisitInfo()

        ha = vi.getBoresightHourAngle().asDegrees()
        airmass = vi.getBoresightAirmass()

        md['HA'] = ha
        md.setComment('HA', 'Hour angle of observation start')

        md['AIRMASS'] = airmass
        md.setComment('AIRMASS', 'Airmass at observation start')

        if 'centroid' in kwargs:
            centroid = kwargs['centroid']
        else:
            centroid = (None, None)

        md['OBJECTX'] = centroid[0]
        md.setComment('OBJECTX', 'x pixel coordinate of object centroid')

        md['OBJECTY'] = centroid[1]
        md.setComment('OBJECTY', 'y pixel coordinate of object centroid')

        exp.setMetadata(md)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs['dataIdDict'] = inputRefs.inputExp.dataId

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def getNormalizedTargetName(self, target):
        """Normalize the name of the target.

        All targets which start with 'spec:' are converted to the name of the
        star without the leading 'spec:'. Any objects with mappings defined in
        data/nameMappings.txt are converted to the mapped name.

        Parameters
        ----------
        target : `str`
            The name of the target.

        Returns
        -------
        normalizedTarget : `str`
            The normalized name of the target.
        """
        target = target.replace('spec:', '')

        with importlib.resources.path("lsst.atmospec", "resources/data/nameMappings.txt") as nameMappingsFile:
            names, mappedNames = np.loadtxt(nameMappingsFile, dtype=str, unpack=True)
        assert len(names) == len(mappedNames)
        conversions = {name: mapped for name, mapped in zip(names, mappedNames)}

        if target in conversions.keys():
            converted = conversions[target]
            self.log.info(f"Converted target name {target} to {converted}")
            return converted
        return target

    def _getSpectractorTargetSetting(self, inputCentroid):
        """Calculate the value to set SPECTRACTOR_FIT_TARGET_CENTROID to.

        Parameters
        ----------
        inputCentroid : `dict`
            The `atmospecCentroid` dict, as received in the task input data.

        Returns
        -------
        centroidMethod : `str`
            The value to set SPECTRACTOR_FIT_TARGET_CENTROID to.
        """

        # if mode is auto and the astrometry worked then it's an exact
        # centroid, and otherwise we fit, as per docs on this option.
        if self.config.targetCentroidMethod == 'auto':
            if inputCentroid['astrometricMatch'] is True:
                self.log.info("Auto centroid is using exact centroid for target from the astrometry")
                return 'guess'  # this means exact
            else:
                self.log.info("Auto centroid is using FIT in Spectractor to get the target centroid")
                return 'fit'  # this means exact

        # this is just renaming the config parameter because guess sounds like
        # an instruction, and really we're saying to take this as given.
        if self.config.targetCentroidMethod == 'exact':
            return 'guess'

        # all other options fall through
        return self.config.targetCentroidMethod

    def run(self, *, inputExp, inputCentroid, dataIdDict):
        if not isDispersedExp(inputExp):
            raise RuntimeError(f"Exposure is not a dispersed image {dataIdDict}")
        starNames = self.loadStarNames()

        overrideDict = {
            # normal config parameters
            'SPECTRACTOR_FIT_TARGET_CENTROID': self._getSpectractorTargetSetting(inputCentroid),
            'SPECTRACTOR_COMPUTE_ROTATION_ANGLE': self.config.rotationAngleMethod,
            'SPECTRACTOR_DECONVOLUTION_PSF2D': self.config.doDeconvolveSpectrum,
            'SPECTRACTOR_DECONVOLUTION_FFM': self.config.doFullForwardModelDeconvolution,
            'SPECTRACTOR_DECONVOLUTION_SIGMA_CLIP': self.config.deconvolutionSigmaClip,
            'SPECTRACTOR_BACKGROUND_SUBTRACTION': self.config.doSubtractBackground,
            'CCD_REBIN': self.config.rebin,
            'XWINDOW': self.config.xWindow,
            'YWINDOW': self.config.yWindow,
            'XWINDOW_ROT': self.config.xWindowRotated,
            'YWINDOW_ROT': self.config.yWindowRotated,
            'PIXSHIFT_PRIOR': self.config.pixelShiftPrior,
            'ROT_PREFILTER': self.config.doFilterRotatedImage,
            'ROT_ORDER': self.config.imageRotationSplineOrder,
            'ROT_ANGLE_MIN': self.config.rotationAngleMin,
            'ROT_ANGLE_MAX': self.config.rotationAngleMax,
            'LINEWIDTH': self.config.plotLineWidth,
            'VERBOSE': self.config.verbose,
            'DEBUG': self.config.spectractorDebugMode,
            'DEBUG_LOGGING': self.config.spectractorDebugLogging,
            'DISPLAY': self.config.doDisplay,
            'LAMBDA_MIN': self.config.lambdaMin,
            'LAMBDA_MAX': self.config.lambdaMax,
            'LAMBDA_STEP': self.config.lambdaStep,
            'SPEC_ORDER': self.config.spectralOrder,
            'PIXWIDTH_SIGNAL': self.config.signalWidth,
            'PIXDIST_BACKGROUND': self.config.backgroundDistance,
            'PIXWIDTH_BACKGROUND': self.config.backgroundWidth,
            'PIXWIDTH_BOXSIZE': self.config.backgroundBoxSize,
            'BGD_ORDER': self.config.backgroundOrder,
            'PSF_TYPE': self.config.psfType,
            'PSF_POLY_ORDER': self.config.psfPolynomialOrder,
            'PSF_FIT_REG_PARAM': self.config.psfRegularization,
            'PSF_PIXEL_STEP_TRANSVERSE_FIT': self.config.psfTransverseStepSize,
            'PSF_FWHM_CLIP': self.config.psfFwhmClip,
            'CALIB_BGD_ORDER': self.config.calibBackgroundOrder,
            'CALIB_PEAK_WIDTH': self.config.calibPeakWidth,
            'CALIB_BGD_WIDTH': self.config.calibBackgroundWidth,
            'CALIB_SAVGOL_WINDOW': self.config.calibSavgolWindow,
            'CALIB_SAVGOL_ORDER': self.config.calibSavgolOrder,
            'OBS_TRANSMISSION_SYSTEMATICS': self.config.transmissionSystematicError,
            'OBS_FULL_INSTRUMENT_TRANSMISSON': self.config.instrumentTransmissionOverride,

            # Hard-coded parameters
            'OBS_NAME': 'AUXTEL',
            'CCD_IMSIZE': 4000,  # short axis - we trim the CCD to square
            'CCD_MAXADU': 170000,  # XXX need to set this from camera value
            'CCD_GAIN': 1.1,  # set programatically later, this is default nominal value
            'OBS_NAME': 'AUXTEL',
            'OBS_ALTITUDE': 2.66299616375123,  # XXX get this from / check with utils value
            'OBS_LATITUDE': -30.2446389756252,  # XXX get this from / check with utils value
            'OBS_EPOCH': "J2000.0",
            'OBS_CAMERA_DEC_FLIP_SIGN': 1,
            'OBS_CAMERA_RA_FLIP_SIGN': 1,
            'OBS_SURFACE': 9636,
            'PAPER': False,
            'SAVE': False,
            'DISTANCE2CCD_ERR': 0.4,

            # Parameters set programatically
            'LAMBDAS': np.arange(self.config.lambdaMin,
                                 self.config.lambdaMax,
                                 self.config.lambdaStep),
            'CALIB_BGD_NPARAMS': self.config.calibBackgroundOrder + 1,

            # Parameters set elsewhere
            # OBS_CAMERA_ROTATION
            # DISTANCE2CCD
        }

        supplementDict = {'CALLING_CODE': 'LSST_DM',
                          'STAR_NAMES': starNames}

        # anything that changes between dataRefs!
        resetParameters = {}
        # TODO: look at what to do with config option doSavePlots

        # TODO: think if this is the right place for this
        # probably wants to go in spectraction.py really
        linearStagePosition = getLinearStagePosition(inputExp)
        _, grating = getFilterAndDisperserFromExp(inputExp)
        if grating == 'holo4_003':
            # the hologram is sealed with a 4 mm window and this is how
            # spectractor handles this, so while it's quite ugly, do this to
            # keep the behaviour the same for now.
            linearStagePosition += 4  # hologram is sealed with a 4 mm window
        overrideDict['DISTANCE2CCD'] = linearStagePosition

        target = inputExp.visitInfo.object
        target = self.getNormalizedTargetName(target)
        if self.config.forceObjectName:
            self.log.info(f"Forcing target name from {target} to {self.config.forceObjectName}")
            target = self.config.forceObjectName

        if target in ['FlatField position', 'Park position', 'Test', 'NOTSET']:
            raise ValueError(f"OBJECT set to {target} - this is not a celestial object!")

        with importlib.resources.path("lsst.atmospec", "resources/config/auxtel.ini") as configFilename:
            spectractor = SpectractorShim(configFile=configFilename,
                                          paramOverrides=overrideDict,
                                          supplementaryParameters=supplementDict,
                                          resetParameters=resetParameters)

        if 'astrometricMatch' in inputCentroid:
            centroid = inputCentroid['centroid']
        else:  # it's a raw tuple
            centroid = inputCentroid  # TODO: put this support in the docstring

        spectraction = spectractor.run(inputExp, *centroid, target,
                                       self.config.doFitAtmosphere,
                                       self.config.doFitAtmosphereOnSpectrogram)

        self.log.info("Finished processing %s" % (dataIdDict))

        return pipeBase.Struct(
            spectractorSpectrum=spectraction.spectrum,
            spectractorImage=spectraction.image,
            spectrumForwardModelFitParameters=spectraction.spectrumForwardModelFitParameters,
            spectrumLibradtranFitParameters=spectraction.spectrumLibradtranFitParameters,
            spectrogramLibradtranFitParameters=spectraction.spectrogramLibradtranFitParameters
        )

    def runAstrometry(self, butler, exp, icSrc):
        refObjLoaderConfig = ReferenceObjectLoader.ConfigClass()
        refObjLoaderConfig.pixelMargin = 1000
        # TODO: needs to be an Input Connection
        refObjLoader = ReferenceObjectLoader(config=refObjLoaderConfig)

        astromConfig = AstrometryTask.ConfigClass()
        astromConfig.wcsFitter.retarget(FitAffineWcsTask)

        # Use magnitude limits for the reference catalog
        astromConfig.referenceSelector.doMagLimit = True
        astromConfig.referenceSelector.magLimit.minimum = 1
        astromConfig.referenceSelector.magLimit.maximum = 15
        astromConfig.referenceSelector.magLimit.fluxField = "phot_g_mean_flux"
        astromConfig.matcher.maxRotationDeg = 5.99
        astromConfig.matcher.maxOffsetPix = 3000

        # Use a SNR limit for the science catalog
        astromConfig.sourceSelector["science"].doSignalToNoise = True
        astromConfig.sourceSelector["science"].signalToNoise.minimum = 10
        astromConfig.sourceSelector["science"].signalToNoise.fluxField = "slot_PsfFlux_instFlux"
        astromConfig.sourceSelector["science"].signalToNoise.errField = "slot_PsfFlux_instFluxErr"
        astromConfig.sourceSelector["science"].doRequirePrimary = False
        astromConfig.sourceSelector["science"].doIsolated = False

        solver = AstrometryTask(config=astromConfig, refObjLoader=refObjLoader)

        # TODO: Change this to doing this the proper way
        referenceFilterName = self.config.referenceFilterOverride
        referenceFilterLabel = afwImage.FilterLabel(physical=referenceFilterName, band=referenceFilterName)
        originalFilterLabel = exp.getFilter()  # there's a better way of doing this with the task I think
        exp.setFilter(referenceFilterLabel)

        try:
            astromResult = solver.run(sourceCat=icSrc, exposure=exp)
            exp.setFilter(originalFilterLabel)
        except (RuntimeError, TaskError):
            self.log.warning("Solver failed to run completely")
            exp.setFilter(originalFilterLabel)
            return None

        scatter = astromResult.scatterOnSky.asArcseconds()
        if scatter < 1:
            return astromResult
        else:
            self.log.warning("Failed to find an acceptable match")
        return None

    def pause(self):
        if self.debug.pauseOnDisplay:
            input("Press return to continue...")
        return

    def loadStarNames(self):
        """Get the objects which should be treated as stars which do not begin
        with HD.

        Spectractor treats all objects which start HD as stars, and all which
        don't as calibration objects, e.g. arc lamps or planetary nebulae.
        Adding items to data/starNames.txt will cause them to be treated as
        regular stars.

        Returns
        -------
        starNames : `list` of `str`
            The list of all objects to be treated as stars despite not starting
            with HD.
        """
        lines = importlib.resources.read_text("lsst.atmospec", "resources/data/starNames.txt").split("\n")
        return [line.strip() for line in lines]

    def flatfield(self, exp, disp):
        """Placeholder for wavelength dependent flatfielding: TODO: DM-18141

        Will probably need a dataRef, as it will need to be retrieving flats
        over a range. Also, it will be somewhat complex, so probably needs
        moving to its own task"""
        self.log.warning("Flatfielding not yet implemented")
        return exp

    def repairCosmics(self, exp, disp):
        self.log.warning("Cosmic ray repair not yet implemented")
        return exp

    def measureSpectrum(self, exp, sourceCentroid, spectrumBBox, dispersionRelation):
        """Perform the spectral extraction, given a source location and exp."""

        self.extraction.initialise(exp, sourceCentroid, spectrumBBox, dispersionRelation)

        # xxx this method currently doesn't return an object - fix this
        spectrum = self.extraction.getFluxBasic()

        return spectrum

    def calcSpectrumBBox(self, exp, centroid, aperture, order='+1'):
        """Calculate the bbox for the spectrum, given the centroid.

        XXX Longer explanation here, inc. parameters
        TODO: Add support for order = "both"
        """
        extent = self.config.spectrumLengthPixels
        halfWidth = aperture//2
        translate_y = self.config.offsetFromMainStar
        sourceX = centroid[0]
        sourceY = centroid[1]

        if order == '-1':
            translate_y = - extent - self.config.offsetFromMainStar

        xStart = sourceX - halfWidth
        xEnd = sourceX + halfWidth - 1
        yStart = sourceY + translate_y
        yEnd = yStart + extent - 1

        xEnd = min(xEnd, exp.getWidth()-1)
        yEnd = min(yEnd, exp.getHeight()-1)
        yStart = max(yStart, 0)
        xStart = max(xStart, 0)
        assert (xEnd > xStart) and (yEnd > yStart)

        self.log.debug('(xStart, xEnd) = (%s, %s)'%(xStart, xEnd))
        self.log.debug('(yStart, yEnd) = (%s, %s)'%(yStart, yEnd))

        bbox = geom.Box2I(geom.Point2I(xStart, yStart), geom.Point2I(xEnd, yEnd))
        return bbox
