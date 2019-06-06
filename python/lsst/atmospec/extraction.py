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
        # xxx if the rest of exp is never used, remove this and just pass exp[spectrumBbox]
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
            except:
                self.log.warn('Failed to initialise debug display')
                self.debug.display = False

        # xxx probably need to change this once per-spectrum background is done
        # xsize = self.spectrumWidth - 2*self.config.perRowBackgroundSize  # 20
        # residuals = np.zeros([xsize, self.spectrumHeight])

        self.backgroundMi = self._calculateBackground(self.footprintExp.maskedImage, 15, smooth=self.config.doSmoothBackround)  # xxx remove hard coding
        self.bgSubMi = self.footprintMi.clone()
        self.bgSubMi -= self.backgroundMi
        if self.debug.display and 'spectrumBgSub' in self.debug.displayItems:
            self.disp1.mtv(self.bgSubMi)

        return

    def _calculateBackground(self, maskedImage, nbins, ignorePlanes=['DETECTED', 'BAD', 'SAT'], smooth=True):
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

        # xxx consider adding a custom mask plane with GROW set high after detection
        # to allow better masking

        bctrl = afwMath.BackgroundControl(nx, ny, sctrl, statType)
        bkgd = afwMath.makeBackground(maskedImage, bctrl)

        bgImg = bkgd.getImageF(afwMath.Interpolate.CONSTANT)

        if not smooth:
            return bgImg

        # bgImg is now an image the same size as the input, as a column of
        # 1 x n blocks, which we now interpolate to make a smooth background
        xs = getSamplePoints(0, bgImg.getHeight()-1, 15, includeEndpoints=True, integers=True)
        vals = [bgImg[0, point, afwImage.LOCAL] for point in xs]

        interpFunc = interp1d(xs, vals, kind='cubic')
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
            pixels = np.arange(0 + self.config.perRowBackgroundSize, (self.spectrumWidth - self.config.perRowBackgroundSize))

        # footprintMi = copy.copy(self.exp[self.spectrumBbox].maskedImage)
        footprintArray = maskedImage.image.array

        residuals = np.zeros([len(pixels), self.spectrumHeight])  # delete this next xxx merlin

        psfAmp = np.max(footprintArray[0])
        psfSigma = np.std(footprintArray[0])
        psfMu = np.argmax(footprintArray[0])
        if abs(psfMu - self.spectrumWidth/2.) >= 10:
            self.log.warn('initial mu more than 10 pixels from footprint center')

        # loop over the rows, calculating basic parameters
        for rowNum in range(self.spectrumHeight):  # take row slices
            if self.config.perRowBackground:
                footprintSlice = self.subtractBkgd(footprintArray[rowNum], self.spectrumWidth, self.config.perRowBackgroundSize)
            else:
                footprintSlice = footprintArray[rowNum]

            # improve this to be a window around max point in row at least
            self.apertureFlux[rowNum] = np.sum(footprintSlice)
            self.rowWiseMax[rowNum] = np.max(footprintSlice)

            if PREVENT_RUNAWAY:
                # xxx values seem odd, probably need changing. Should be independent of width
                # xxx also should check if redoing this each time is better/worse
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
                              bounds=(0., [100*np.max(footprintSlice), self.spectrumWidth, 2*self.spectrumWidth]))
                psfFlux = np.sqrt(2*np.pi) * psfSigma * psfAmp  # Gaussian integral
                # parErr = np.sqrt(np.diag(varMatrix))
                self.psfFitPars[rowNum] = (psfAmp, psfMu, psfSigma, psfFlux)

            except (RuntimeError, ValueError) as e:
                # serious, and should never happen
                self.log.warn(f'\nRuntimeError for basic 1D Gauss fit! rowNum = {rowNum}\n{e}')

            try:
                fitValsMoffat = self.moffatFit(pixels, footprintSlice, psfAmp, psfMu, psfSigma,
                                               residuals, rowNum)
                self.moffatFitPars[rowNum] = fitValsMoffat
            except (RuntimeError, ValueError):
                self.log.warn(f'\n\nRuntimeError during Moffat fit! rowNum = {rowNum}\n')  # serious, and should never happen

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

                fitValsGM = self.gaussMoffatFit(pixels, footprintSlice, initialPars,
                                                residuals, rowNum)

                self.gausMoffatFitPars[rowNum] = fitValsGM

            except (RuntimeError, ValueError):
                msg = f'\n\nRuntimeError during GaussMoffatFit fit! rowNum = {rowNum}\n'
                self.log.warn(msg)  # serious, and should never happen
                # self.gausMoffatFitPars.append(fitValsGM)

            '''Filling in the residuals'''
            # fit = self.gauss1D(pixels, *coeffs)
            # residuals[:, rowNum] = fit-footprintSlice #currently at Moffat

            if self.debug.plot and ('all' in self.debug.plot or 'GausMoffat' in self.debug.plot):
                # if((not rowNum % 10) and (rowNum < 400)):
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
            # hdu = pf.PrimaryHDU(residuals)
            # name = "residuals.fits"
            # hdu.writeto(os.path.join(self.spectrum.out_dir, name), overwrite=True)
            # logging.info("writing map of residuals")

        return

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
    def gauss1D(x, *p):
        amp, mu, sigma = p
        return amp*np.exp(-(x-mu)**2/(2.*sigma**2))

    @staticmethod
    def moffatFit(pixels, footprint, A, mu, sigma, residuals, i):
        initialMoffat = models.Moffat1D(amplitude=A, x_0=mu, gamma=sigma)
        fitter = fitting.LevMarLSQFitter()
        psf = fitter(initialMoffat, pixels, footprint)

        start = psf.x_0 - 5 * psf.gamma
        end = psf.x_0 + 5 * psf.gamma
        integral = (integrate.quad(lambda pixels: psf(pixels), start, end))[0]

        # if((not i % 10) and (i < 400) and (plot == True)):
        #     pl.plot(pixels, psf(pixels), label='Moffat')
        #     pl.yscale('log')
        #     pl.ylim(1., 1E6)
        #     pl.plot(pixels, footprint)
        #     pl.legend()
        #     pl.show()

        '''Filling residuals'''
        # residuals[:, i] = psf(pixels)-footprint
        return integral, psf.x_0.value, psf.gamma.value, psf.alpha.value

    @staticmethod
    def gaussMoffatFit(pixels, footprint, initialPars, residuals, rowNum):
        moffatAmp, x_0, gamma, alpha, gausAmp, gausMu, gausSigma = initialPars
    # def gaussMoffatFit(pixels, footprint, amplitude, x_0, gamma, alpha, A, gausMu, gausSigma,
    # residuals, rowNum):

        initialMoffat = models.Moffat1D(amplitude=moffatAmp, x_0=x_0, gamma=gamma, alpha=alpha)
        initialMoffat.amplitude.min = 0.
        initialMoffat.x_0.min = min(footprint)-5
        initialMoffat.x_0.max = max(pixels)+5
        # initialMoffat.amplitude.max = gausAmp/10.
        initialMoffat.gamma.min = 1.
        initialMoffat.gamma.max = 2.
        initialMoffat.alpha.min = 1.
        initialMoffat.alpha.max = 2.

        initialGaus = models.Gaussian1D(amplitude=gausAmp, mean=gausMu, stddev=gausSigma)
        initialGaus.amplitude.min = 0.
        initialGaus.mean.min = min(footprint)-5
        initialGaus.mean.max = max(footprint)+5
        initialGaus.stddev.min = 0.5
        initialGaus.stddev.max = 5.

        initialModel = initialMoffat + initialGaus

        fitter = fitting.LevMarLSQFitter()

        psf = fitter(initialModel, pixels, footprint)

        start = psf.x_0_0 - 10 * psf.stddev_1
        end = psf.x_0_0 + 10 * psf.stddev_1
        intGM = (integrate.quad(lambda pixels: psf(pixels), start, end))[0]
        intG = np.sqrt(2 * np.pi) * psf.stddev_1 * psf.amplitude_1

        # if((not i % 10)and(i < 400) and (plot == True)):
        #     pl.plot(pixels, psf(pixels), label='Gauss+Moffat')
        #     pl.yscale('log')
        #     pl.ylim(1., 1E6)
        #     pl.plot(pixels, footprint)
        #     pl.legend()
        #     pl.show()

        residuals[:, rowNum] = psf(pixels)-footprint

        fitGausAmp = psf.amplitude_0.value
        fitX0 = psf.x_0_0.value
        fitGam = psf.gamma_0.value
        fitAlpha = psf.alpha_0.value
        fitMofAmp = psf.amplitude_1.value
        fitMofMean = psf.mean_1.value
        fitMofWid = psf.stddev_1.value
        return intGM, intG, fitGausAmp, fitX0, fitGam, fitAlpha, fitMofAmp, fitMofMean, fitMofWid
