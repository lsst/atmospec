import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import logging

from scipy.ndimage import median_filter, uniform_filter, gaussian_filter

"""
The goal of this module is to provide a simple interface to build a flat adapted to
empty~holo4_003 AuxTel exposures. The more recent empty~empty AuxTel certified flat
is found for a given empty~holo4_003 AuxTel exposure. It is then 

"""


def findFlats(butler, cameraName='LATISS', filter='empty', disperser='empty', obsType='flat'):
    """Find the certified flats for a given instrument and set of filters

    :param butler:
    :param cameraName:
    :param filter:
    :param disperser:
    :param obsType:
    :return:
    """
    registry = butler.registry
    physicalFilter = f'{filter}~{disperser}'
    where = (
        f"instrument='{cameraName}' AND physical_filter='{physicalFilter}' "
        f"AND exposure.observation_type='{obsType}'"
    )
    flatRecords = list(registry.queryDimensionRecords('exposure', where=where))

    if len(flatRecords) == 0:
        raise ValueError(
            'WARNING: No flats found with the selected settings: \n'
            f'{cameraName=} {physicalFilter=} {obsType=}')

    flatDates = np.sort(np.unique([r.day_obs for r in flatRecords]))
    ids = [r.id for r in flatRecords]

    flatDictIds = {}
    for date in flatDates:
        flatDictIds[date] = []
        for expId in ids:
            if str(date) in str(expId):
                flatDictIds[date].append(expId)
        flatDictIds[date] = np.sort(flatDictIds[date])

    return flatDates, flatDictIds


def getCertifiedFlat(butler, dataId, filter='empty', disperser='empty'):
    """Get the certified flats for a given instrument and set of filters the most recent
    considering the provided data Id.

    :param butler:
    :param dataId:
    :param filter:
    :param disperser:
    :return:
    """
    flatDates, flatIds = findFlats(butler, filter=filter, disperser=disperser)
    dayObs = dataId // 100_000
    dateDiff = int(dayObs) - flatDates
    closestDate = flatDates[np.argmax(dateDiff[dateDiff <= 0])]
    # load flat
    flatId = flatIds[closestDate][-1]
    certifiedFlat = butler.get('flat', instrument='LATISS', exposure=flatId, detector=0,
                               collections=['LATISS/calib', 'LATISS/raw/all'])
    return certifiedFlat


def getPTCGainDict(butler):
    """Get the PTC gains from a LATISS collection.

    :param butler:
    :return:
    """
    ptc = butler.get('ptc', instrument="LATISS", detector=0, collections='u/cslage/sdf/latiss/ptc_20220927J')
    ptcGainDict = ptc.gain
    return ptcGainDict


def makeGainFlat(certifiedFlat, gainDict, invertGains=False):
    """Given an exposure, make a flat from the gains.

    Construct an exposure where the image array data contains the gain of the
    pixel, such that dividing by (or mutiplying by) the image will convert
    an image from ADU to electrons.

    Parameters
    ----------
    certifiedFlat : `lsst.afw.image.exposure`
        The certifiedFlat from which the flat is to be made.

    gainDict : `dict` of `float`
        A dict of the amplifiers' gains, keyed by the amp names.

    invertGains : `bool`
        Gains are specified in inverted units and should be applied as such.

    Returns
    -------
    gainFlat : `lsst.afw.image.exposure`
        The gain flat
    """
    flat = certifiedFlat.clone()
    detector = flat.getDetector()
    ampNames = set(list(a.getName() for a in detector))
    assert set(gainDict.keys()) == ampNames

    gainDictNorm = {}
    mean = np.mean(list(gainDict.values()))
    for amp in gainDict.keys():
        gainDictNorm[amp] = gainDict[amp] / mean

    for amp in detector:
        bbox = amp.getBBox()
        if invertGains:
            flat[bbox].maskedImage.image.array[:, :] = 1. / gainDictNorm[amp.getName()]
        else:
            flat[bbox].maskedImage.image.array[:, :] = gainDictNorm[amp.getName()]
    flat.maskedImage.mask[:] = 0x0
    flat.maskedImage.variance[:] = 0.0

    return flat


def makeOpticFlat(certifiedFlat, kernel='mean', windowSize=40, mode='mirror', percentile=1.):
    """Build an Optic Flat from a certified Flat.
     An Optic Flat contains only the large-scale defects coming from the optics.
     The different amplifiers are rescaled to median=1, and small scale defects are filtered.
     If kernel='mean' filter is used, outlier pixels below the percentile threshold are filtered out.

    :param certifiedFlat:
    :param kernel:
    :param windowSize:
    :param mode:
    :param percentile:
    :return:
    """
    logger = logging.getLogger()
    opticFlat = certifiedFlat.clone()

    detector = opticFlat.getDetector()
    for amp in detector:
        bbox = amp.getBBox()
        data = np.copy(opticFlat[bbox].maskedImage.image.array[:, :])
        data /= np.median(data)
        if kernel == 'gauss':
            logger.info(f'Pixel flat smoothing with Gaussian filter with {windowSize=}.')
            # 'ATTENTION: window size should be equal to Gaussian standard
            # deviation (sigma)'
            opticFlat[bbox].maskedImage.image.array[:, :] = gaussian_filter(data, sigma=windowSize,
                                                                            mode=mode)
        elif kernel == 'mean':
            logger.info(
                f'Pixel flat smoothing with mean filter.\n'
                f'Masking outliers <{percentile:.2f} and >{100. - percentile:.2f} percentiles'
            )
            mask = (
                data >= np.percentile(data.flatten(), percentile)) * (
                data <= np.percentile(data.flatten(), 100. - percentile)
            )
            interpArray = np.copy(data)
            interpArray[~mask] = np.median(data)
            opticFlat[bbox].maskedImage.image.array[:, :] = uniform_filter(interpArray, size=windowSize,
                                                                           mode=mode)
        elif kernel == 'median':
            logger.info(f'Pixel flat smoothing with median filter with {windowSize=}.')
            opticFlat[bbox].maskedImage.image.array[:, :] = median_filter(data,
                                                                          size=(windowSize, windowSize),
                                                                          mode=mode)
        else:
            raise ValueError(f'Kernel must be within ["mean", "median", "gauss"]. Got {kernel=}.')

    opticFlat.maskedImage.mask[:] = 0x0
    opticFlat.maskedImage.variance[:] = 0.0
    return opticFlat


def makePixelFlat(
    certifiedFlat,
    kernel='mean',
    windowSize=40,
    mode='mirror',
    percentile=1.,
    removeBackground=True
):
    """Build a Pixel Flat from a certified Flat.
     A Pixel Flat contains only the defects coming from the pixel efficiencies.

    :param certifiedFlat:
    :param kernel:
    :param windowSize:
    :param mode:
    :param percentile:
    :param removeBackground:
    :return:
    """
    logger = logging.getLogger()
    logger.info(f'Window size for {kernel} smoothing = {windowSize}')
    pixelFlat = certifiedFlat.clone()
    opticFlat = makeOpticFlat(certifiedFlat, kernel=kernel, windowSize=windowSize, mode=mode,
                              percentile=percentile)

    detector = opticFlat.getDetector()
    for amp in detector:
        bbox = amp.getBBox()
        ampData = pixelFlat[bbox].maskedImage.image.array[:, :]
        ampData /= np.median(ampData)
        ampData /= opticFlat[bbox].maskedImage.image.array[:, :]
        if removeBackground:
            from astropy.stats import SigmaClip
            from photutils.background import Background2D, MedianBackground
            sigma_clip = SigmaClip(sigma=3.0)
            bkg_estimator = MedianBackground()
            bkg = Background2D(ampData, (20, 20),
                               filter_size=(3, 3),
                               sigma_clip=sigma_clip,
                               bkg_estimator=bkg_estimator)
            ampData /= bkg.background

    pixelFlat.maskedImage.mask[:] = 0x0
    pixelFlat.maskedImage.variance[:] = 0.0
    return pixelFlat

def makeSensorFlat(
    certifiedFlat,
    gainDict,
    kernel='mean',
    windowSize=40,
    mode='mirror',
    percentile=1.,
    invertGains=False,
    removeBackground=True
):
    """Build a Sensor Flat from a certified Flat.
     A Sensor Flat contains only the defects coming from the Sensor (pixel efficiencies, amplifier gains).

    :param certifiedFlat:
    :param gainDict:
    :param kernel:
    :param windowSize:
    :param mode:
    :param percentile:
    :param invertGains:
    :param removeBackground:
    :return:
    """
    logger = logging.getLogger()
    logger.info(f'Window size for {kernel} smoothing = {windowSize}')
    pixelFlat = makePixelFlat(certifiedFlat, kernel=kernel, windowSize=windowSize, mode=mode,
                              percentile=percentile, removeBackground=removeBackground)
    gainFlat = makeGainFlat(certifiedFlat, gainDict, invertGains=invertGains)
    sensorFlat = pixelFlat.clone()

    detector = sensorFlat.getDetector()
    ampNames = set(list(a.getName() for a in detector))
    assert set(gainDict.keys()) == ampNames

    for amp in detector:
        bbox = amp.getBBox()
        sensorFlat[bbox].maskedImage.image.array[:, :] *= gainFlat[bbox].maskedImage.image.array[:, :]

    sensorFlat.maskedImage.mask[:] = 0x0
    sensorFlat.maskedImage.variance[:] = 0.0
    return sensorFlat


def plotFlat(flat, title=None, figsize=(10, 10), cmap='gray', vmin=0.9, vmax=1.1, lognorm=False):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if lognorm:
        im = ax.imshow(flat.image.array, cmap=cmap, origin='lower', norm=LogNorm())
    else:
        im = ax.imshow(flat.image.array, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    if title is not None:
        ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return
