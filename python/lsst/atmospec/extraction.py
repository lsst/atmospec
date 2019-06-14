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

import numpy as np
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.interpolate import interp1d
from astropy.modeling import models, fitting
import matplotlib.pyplot as pl

import lsstDebug
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
from .utils import getSamplePoints, argMaxNd


__all__ = ['SpectralExtractionTask', 'SpectralExtractionTaskConfig']

PREVENT_RUNAWAY = False


class SpectralExtractionTaskConfig(pexConfig.Config):
    perRowBackground = pexConfig.Field(
        doc="If True, subtract backgroud per-row, else subtract for whole image",
        dtype=bool,
        default=False,
    )
    perRowBackgroundSize = pexConfig.Field(
        doc="Background box size (or width, if perRowBackground)",
        dtype=int,
        default=10,
    )
    writeResiduals = pexConfig.Field(
        doc="Background box size (or width, if perRowBackground)",
        dtype=bool,
        default=False,
    )
    doSmoothBackround = pexConfig.Field(
        doc="Spline smooth the 1 x n background boxes?",
        dtype=bool,
        default=True,
    )
    doSigmaClipBackground = pexConfig.Field(
        doc="Sigma clip the background model in addition to masking detected (and other) pixels?",
        dtype=bool,
        default=False
    )
    nSigmaClipBackground = pexConfig.Field(
        doc="Number of sigma to clip to if appying sigma clipping to background model",
        dtype=float,
        default=5
    )
    nSigmaClipBackgroundIterations = pexConfig.Field(
        doc="Number of iterations if appying sigma clipping to background model",
        dtype=int,
        default=4
    )


class SpectralExtractionTask(pipeBase.Task):

    ConfigClass = SpectralExtractionTaskConfig
    _DefaultName = "spectralExtraction"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.debug = lsstDebug.Info(__name__)

    def initialise(self, exp, sourceCentroid, spectrumBbox, dispersionRelation):
        """xxx Docstring here.

        Parameters:
        -----------
        par : `type
            Description

        """
        # xxx if the rest of exp is never used, remove this
        # and just pass exp[spectrumBbox]
        self.expRaw = exp
        self.footprintExp = exp[spectrumBbox]
        self.footprintMi = self.footprintExp.maskedImage
        self.sourceCentroid = sourceCentroid
        self.spectrumBbox = spectrumBbox
        self.dispersionRelation = dispersionRelation
        self.spectrumWidth = self.spectrumBbox.getWidth()
        self.spectrumHeight = self.spectrumBbox.getHeight()

        # profiles - aperture and psf
        self.apertureFlux = np.zeros(self.spectrumHeight)
        self.rowWiseMax = np.zeros(self.spectrumHeight)
        # self.psfSigma = np.zeros(self.spectrumHeight)
        # self.psfMu = np.zeros(self.spectrumHeight)
        # self.psfFlux = np.zeros(self.spectrumHeight)

        # profiles - Moffat
        # self.moffatFlux = np.zeros(self.spectrumHeight)
        # self.moffatX0 = np.zeros(self.spectrumHeight)
        # self.moffatGamma = np.zeros(self.spectrumHeight)
        # self.moffatAlpha = np.zeros(self.spectrumHeight)

        # profiles - Gauss + Moffat
        # self.gmIntegralGM = np.zeros(self.spectrumHeight)
        # self.gmIntegralG = np.zeros(self.spectrumHeight)

        # probably delete these - from the GausMoffat fitting
        # self.amplitude_0 = np.zeros(self.spectrumHeight)
        # self.x_0_0 = np.zeros(self.spectrumHeight)
        # self.gamma_0 = np.zeros(self.spectrumHeight)
        # self.alpha_0 = np.zeros(self.spectrumHeight)
        # self.amplitude_1 = np.zeros(self.spectrumHeight)
        # self.mean_1 = np.zeros(self.spectrumHeight)
        # self.stddev_1 = np.zeros(self.spectrumHeight)

        # each will be an list of length self.spectrumHeight
        # with each element containing all the fit parameters
        self.psfFitPars = [None] * self.spectrumHeight
        self.moffatFitPars = [None] * self.spectrumHeight
        self.gausMoffatFitPars = [None] * self.spectrumHeight

        if self.debug.display:
            try:
                import lsst.afw.display as afwDisp
                afwDisp.setDefaultBackend(self.debug.displayBackend)
                # afwDisp.Display.delAllDisplays()
                self.disp1 = afwDisp.Display(0, open=True)
                self.disp1.mtv(self.expRaw[self.spectrumBbox])
                self.disp1.erase()
            except Exception:
                self.log.warn('Failed to initialise debug display')
                self.debug.display = False

        # xxx probably need to change this once per-spectrum background is done
        # xsize = self.spectrumWidth - 2*self.config.perRowBackgroundSize  # 20
        # residuals = np.zeros([xsize, self.spectrumHeight])

        self.backgroundMi = self._calculateBackground(self.footprintExp.maskedImage,   # xxx remove hardcoding
                                                      15, smooth=self.config.doSmoothBackround)
        self.bgSubMi = self.footprintMi.clone()
        self.bgSubMi -= self.backgroundMi
        if self.debug.display and 'spectrumBgSub' in self.debug.displayItems:
            self.disp1.mtv(self.bgSubMi)

        return

    def _calculateBackground(self, maskedImage, nbins, ignorePlanes=['DETECTED', 'BAD', 'SAT'], smooth=True):

        assert nbins > 0
        if nbins > maskedImage.getHeight() - 1:
            self.log.warn(f"More bins selected for background than pixels in image height." +
                          f" Reducing numbers of bins from {nbins} to {maskedImage.getHeight()-1}.")
            nbins = maskedImage.getHeight() - 1

        nx = 1
        ny = nbins
        sctrl = afwMath.StatisticsControl()
        MaskPixel = afwImage.MaskPixel
        sctrl.setAndMask(afwImage.Mask[MaskPixel].getPlaneBitMask(ignorePlanes))

        if self.config.doSigmaClipBackground:
            sctrl.setNumSigmaClip(self.config.nSigmaClipBackground)
            sctrl.setNumIter(self.config.nSigmaClipBackgroundIterations)
            statType = afwMath.MEANCLIP
        else:
            statType = afwMath.MEAN

        # xxx consider adding a custom mask plane with GROW set high after
        # detection to allow better masking

        bctrl = afwMath.BackgroundControl(nx, ny, sctrl, statType)
        bkgd = afwMath.makeBackground(maskedImage, bctrl)

        bgImg = bkgd.getImageF(afwMath.Interpolate.CONSTANT)

        if not smooth:
            return bgImg

        # Note that nbins functions differently for scipy.interp1d than for
        # afwMath.BackgroundControl

        nbins += 1  # if you want 1 bin you must specify two to getSamplePoints() because of ends

        kind = 'cubic' if nbins >= 4 else 'linear'  # note nbinbs has been incrememented

        # bgImg is now an image the same size as the input, as a column of
        # 1 x n blocks, which we now interpolate to make a smooth background
        xs = getSamplePoints(0, bgImg.getHeight()-1, nbins, includeEndpoints=True, integers=True)
        vals = [bgImg[0, point, afwImage.LOCAL] for point in xs]

        interpFunc = interp1d(xs, vals, kind=kind)
        xsNew = np.linspace(0, bgImg.getHeight()-1, bgImg.getHeight())  # integers over range
        assert xsNew[1]-xsNew[0] == 1  # easy to make off-by-one errors here

        interpVals = interpFunc(xsNew)

        for col in range(bgImg.getWidth()):  # there must be a more pythonic way to tile these into the image
            bgImg.array[:, col] = interpVals

        return bgImg

    def getFluxBasic(self):
        """Docstring here."""

        # xxx check if this is modified and dispose of copy if not
        # footprint = self.exp[self.spectrumBbox].maskedImage.image.array.copy()

        if not self.config.perRowBackground:
            maskedImage = self.bgSubMi
            pixels = np.arange(self.spectrumWidth)
        else:
            maskedImage = self.footprintMi
            pixels = np.arange(0 + self.config.perRowBackgroundSize,
                               (self.spectrumWidth - self.config.perRowBackgroundSize))

        # footprintMi = copy.copy(self.exp[self.spectrumBbox].maskedImage)
        footprintArray = maskedImage.image.array

        # delete this next xxx merlin
        # residuals = np.zeros([len(pixels), self.spectrumHeight])

        psfAmp = np.max(footprintArray[0])
        psfSigma = np.std(footprintArray[0])
        psfMu = np.argmax(footprintArray[0])
        if abs(psfMu - self.spectrumWidth/2.) >= 10:
            self.log.warn('initial mu more than 10 pixels from footprint center')

        # loop over the rows, calculating basic parameters
        for rowNum in range(self.spectrumHeight):  # take row slices
            if self.config.perRowBackground:
                footprintSlice = self.subtractBkgd(footprintArray[rowNum], self.spectrumWidth,
                                                   self.config.perRowBackgroundSize)
            else:
                footprintSlice = footprintArray[rowNum]

            # improve this to be a window around max point in row at least
            self.apertureFlux[rowNum] = np.sum(footprintSlice)
            self.rowWiseMax[rowNum] = np.max(footprintSlice)

            if PREVENT_RUNAWAY:
                # xxx values seem odd, probably need changing.
                # xxx Should be independent of width. Also should check if
                # redoing this each time is better/worse
                # if so then psfAmp = np.max(footprintArray[0])
                if ((psfMu > .7*self.spectrumWidth) or (psfMu < 0.3*self.spectrumWidth)):
                    # psfMu = width/2.  # Augustin's method
                    psfMu = np.argmax(footprintSlice)
                    self.log.warn(f'psfMu was running away on row {rowNum} - reset to nominal')
                if ((psfSigma > 20.) or (psfSigma < 0.1)):
                    # psfSigma = 3.  # Augustin's method
                    psfSigma = np.std(footprintSlice)
                    self.log.warn(f'psfSigma was running away on row {rowNum} - reset to nominal')

            initialPars = [psfAmp, psfMu, psfSigma]  # use value from previous iteration

            try:
                (psfAmp, psfMu, psfSigma), varMatrix = \
                    curve_fit(self.gauss1D, pixels, footprintSlice, p0=initialPars,
                              bounds=(0., [100*np.max(footprintSlice),
                                      self.spectrumWidth, 2*self.spectrumWidth]))
                psfFlux = np.sqrt(2*np.pi) * psfSigma * psfAmp  # Gaussian integral
                # parErr = np.sqrt(np.diag(varMatrix))
                self.psfFitPars[rowNum] = (psfAmp, psfMu, psfSigma, psfFlux)

            except (RuntimeError, ValueError) as e:
                self.log.warn(f'\nRuntimeError for basic 1D Gauss fit! rowNum = {rowNum}\n{e}')

            try:
                fitValsMoffat = self.moffatFit(pixels, footprintSlice, psfAmp, psfMu, psfSigma)
                self.moffatFitPars[rowNum] = fitValsMoffat
            except (RuntimeError, ValueError) as e:
                self.log.warn(f'\n\nRuntimeError during Moffat fit! rowNum = {rowNum}\n{e}')

            try:
                if rowNum == 0:  # bootstrapping, hence all the noqa: F821
                    initialPars = (np.max(footprintSlice),
                                   np.argmax(footprintSlice),
                                   0.5,
                                   0.5,
                                   self.psfFitPars[rowNum][3]/2,
                                   np.argmax(footprintSlice),
                                   0.5)
                elif self.psfFitPars[rowNum] is None:  # basic PSF fitting failed!
                    initialPars = (np.sum(footprintSlice),
                                   fitValsGM[3],  # noqa: F821
                                   fitValsGM[4],  # noqa: F821
                                   fitValsGM[5],  # noqa: F821
                                   fitValsGM[6],  # noqa: F821
                                   fitValsGM[7],  # noqa: F821
                                   fitValsGM[8])  # noqa: F821
                else:
                    initialPars = (self.psfFitPars[rowNum][3],
                                   fitValsGM[3],  # noqa: F821
                                   fitValsGM[4],  # noqa: F821
                                   fitValsGM[5],  # noqa: F821
                                   fitValsGM[6],  # noqa: F821
                                   fitValsGM[7],  # noqa: F821
                                   fitValsGM[8])  # noqa: F821

                fitValsGM = self.gaussMoffatFit(pixels, footprintSlice, initialPars)

                self.gausMoffatFitPars[rowNum] = fitValsGM

            except (RuntimeError, ValueError) as e:
                msg = f'\n\nRuntimeError during GaussMoffatFit fit! rowNum = {rowNum}\n{e}'
                self.log.warn(msg)  # serious, and should never happen
                # self.gausMoffatFitPars.append(fitValsGM)

            '''Filling in the residuals'''
            # fit = self.gauss1D(pixels, *coeffs)
            # residuals[:, rowNum] = fit-footprintSlice #currently at Moffat

            if self.debug.plot and ('all' in self.debug.plot or 'GausMoffat' in self.debug.plot):
                # if((not rowNum % 10) and (rowNum < 400)):
                if True:
                    print('aper : ', rowNum, self.apertureFlux[rowNum])
                    print(rowNum, self.psf_gauss_flux[rowNum], self.psf_gauss_psfSigma[rowNum],
                          self.psf_gauss_psfMu[rowNum], psfAmp)
                    # fit = Gauss1D(pixels, *coeffs)
                    pl.xlabel('Spectrum spatial profile (pixel)')
                    pl.ylabel('Amplitude (ADU)')
                    pl.title('CTIO .9m - %s'%(self.spectrum.object_name))
                    pl.plot(pixels, fit, label='Gauss')
                    pl.yscale('log')
                    pl.ylim(1., 1E6)
                    pl.plot(pixels, footprintSlice)
                    pl.legend()
                    pl.show()

        ##########################################################
        # after the row-by-row processing
        maxRow = argMaxNd(footprintArray)[0]

        if self.config.writeResiduals:
            self.log.warn('Not implemented yet')

        return self

    @staticmethod
    def subtractBkgd(slice, width, bkgd_size):
        # xxx remove these slices - seems totally unnecessary
        subFootprint = slice[bkgd_size:(-1-bkgd_size+1)]
        bkgd1 = slice[:bkgd_size]
        bkgd2 = slice[width-bkgd_size:]
        bkgd1 = np.median(bkgd1)
        bkgd2 = np.median(bkgd2)
        bkgd = np.mean([bkgd1, bkgd2])
        subFootprint = subFootprint - bkgd
        return subFootprint

    @staticmethod
    def gauss1D(x, *pars):
        amp, mu, sigma = pars
        return amp*np.exp(-(x-mu)**2/(2.*sigma**2))

    @staticmethod
    def moffatFit(pixels, footprint, amp, mu, sigma):
        pars = (amp, mu, sigma)
        initialMoffat = moffatModel(pars)
        fitter = fitting.LevMarLSQFitter()
        mof = fitter(initialMoffat, pixels, footprint)

        start = mof.x_0 - 5 * mof.gamma
        end = mof.x_0 + 5 * mof.gamma
        integral = (integrate.quad(lambda pixels: mof(pixels), start, end))[0]

        return integral, mof.x_0.value, mof.gamma.value, mof.alpha.value

    @staticmethod
    def gaussMoffatFit(pixels, footprint, initialPars):
        model = gausMoffatModel(initialPars)

        model.amplitude_1.min = 0.
        model.x_0_1.min = min(footprint)-5
        model.x_0_1.max = max(pixels)+5
        # model.amplitude_1.max = gausAmp/10.
        model.gamma_1.min = 1.
        model.gamma_1.max = 2.
        model.alpha_1.min = 1.
        model.alpha_1.max = 2.

        model.amplitude_0.min = 0.
        model.mean_0.min = min(footprint)-5
        model.mean_0.max = max(footprint)+5
        model.stddev_0.min = 0.5
        model.stddev_0.max = 5.

        fitter = fitting.LevMarLSQFitter()
        psf = fitter(model, pixels, footprint)

        start = psf.x_0_1 - 10 * psf.stddev_0
        end = psf.x_0_1 + 10 * psf.stddev_0
        intGM = (integrate.quad(lambda pixels: psf(pixels), start, end))[0]
        intG = np.sqrt(2 * np.pi) * psf.stddev_0 * psf.amplitude_0

        fitGausAmp = psf.amplitude_1.value
        fitX0 = psf.x_0_1.value
        fitGam = psf.gamma_1.value
        fitAlpha = psf.alpha_1.value
        fitMofAmp = psf.amplitude_0.value
        fitMofMean = psf.mean_0.value
        fitMofWid = psf.stddev_0.value
        return intGM, intG, fitGausAmp, fitX0, fitGam, fitAlpha, fitMofAmp, fitMofMean, fitMofWid


def gausMoffatModel(pars):
    moffatAmp, x0, gamma, alpha, gausAmp, mu, gausSigma = pars
    moff = models.Moffat1D(amplitude=moffatAmp, x_0=x0, gamma=gamma, alpha=alpha)
    gaus = models.Gaussian1D(amplitude=gausAmp, mean=mu, stddev=gausSigma)
    model = gaus + moff
    return model


def moffatModel(pars):
    amp, mu, sigma = pars
    moff = models.Moffat1D(amplitude=amp, x_0=mu, gamma=sigma)
    return moff
