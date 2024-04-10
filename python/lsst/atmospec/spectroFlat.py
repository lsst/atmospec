import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import logging

from scipy.ndimage import median_filter, uniform_filter, gaussian_filter


# Butler
# import lsst.daf.butler as dafButler
# butler = dafButler.Butler(repo, collections=calibCollections)


def find_flats(butler, cameraName='LATISS', filter='empty', disperser='empty', obs_type='flat'):
    registry = butler.registry
    physical_filter = f'{filter}~{disperser}'
    where = f"instrument='{cameraName}' AND physical_filter='{physical_filter}' AND exposure.observation_type='{obs_type}'"
    flat_records = list(registry.queryDimensionRecords('exposure', where=where))

    if len(flat_records) == 0:
        raise ValueError(
            'WARNING: No flats found with the selected settings: \n{cameraName=} {physical_filter=} {obs_type=}')

    flat_dates = np.sort(np.unique([flat_.day_obs for flat_ in flat_records]))
    ids_ = [flat_.id for flat_ in flat_records]

    flat_dict_ids = {}
    for date_ in flat_dates:
        flat_dict_ids[date_] = []
        for id_ in ids_:
            if str(date_) in str(id_):
                flat_dict_ids[date_].append(id_)
        flat_dict_ids[date_] = np.sort(flat_dict_ids[date_])

    return flat_dates, flat_dict_ids


def getCertifiedFlat(butler, dataId, filter='empty', disperser='empty'):
    flat_dates, flat_ids = find_flats(butler, filter=filter, disperser=disperser)
    obs_day = dataId // 100_000
    date_diff = int(obs_day) - flat_dates
    closest_date = flat_dates[np.argmax(date_diff[date_diff <= 0])]
    # load flat
    flat_id = flat_ids[closest_date][-1]
    certifiedFlat = butler.get('flat', instrument='LATISS', exposure=flat_id, detector=0,
                               collections=['LATISS/calib', 'LATISS/raw/all'])
    return certifiedFlat


def getGainDict(exposure):
    det = exposure.getDetector()
    gainDict = {}
    for amp in det:
        gainDict[amp.getName()] = amp.getGain()
    return gainDict


def getPTCGainDict(butler):
    ptc = butler.get('ptc', instrument="LATISS", detector=0, collections='u/cslage/sdf/latiss/ptc_20220927J')
    PTCGainDict = ptc.gain
    return PTCGainDict


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

    gainDict_norm = {}
    mean = np.mean(list(gainDict.values()))
    for amp in gainDict.keys():
        gainDict_norm[amp] = gainDict[amp] / mean

    for amp in detector:
        bbox = amp.getBBox()
        if invertGains:
            flat[bbox].maskedImage.image.array[:, :] = 1. / gainDict_norm[amp.getName()]
        else:
            flat[bbox].maskedImage.image.array[:, :] = gainDict_norm[amp.getName()]
    flat.maskedImage.mask[:] = 0x0
    flat.maskedImage.variance[:] = 0.0

    return flat


def makeOpticFlat(certifiedFlat, kernel='mean', window_size=40, mode='mirror', percentile=1.):
    logger = logging.getLogger()
    opticFlat = certifiedFlat.clone()

    detector = opticFlat.getDetector()
    for amp in detector:
        bbox = amp.getBBox()
        data = np.copy(opticFlat[bbox].maskedImage.image.array[:, :])
        data /= np.median(data)
        if kernel == 'gauss':
            logger.info(f'Pixel flat smoothing with Gaussian filter with {window_size=}.')
            # 'ATTENTION: window size should be equal to Gaussian standard deviation (sigma)'
            opticFlat[bbox].maskedImage.image.array[:, :] = gaussian_filter(data, sigma=window_size, mode=mode)
        elif kernel == 'mean':
            logger.info(
                f'Pixel flat smoothing with mean filter.\nMasking outliers <{percentile:.2f} and >{100. - percentile:.2f} percentiles')
            mask = (data >= np.percentile(data.flatten(), percentile)) * (
                        data <= np.percentile(data.flatten(), 100. - percentile))
            interp_array = np.copy(data)
            interp_array[~mask] = np.median(data)
            opticFlat[bbox].maskedImage.image.array[:, :] = uniform_filter(interp_array, size=window_size, mode=mode)
        elif kernel == 'median':
            logger.info(f'Pixel flat smoothing with median filter with {window_size=}.')
            opticFlat[bbox].maskedImage.image.array[:, :] = median_filter(data, size=(window_size, window_size),
                                                                          mode=mode)
        else:
            raise ValueError(f'Kernel must be within ["mean", "median", "gauss"]. Got {kernel=}.')

    opticFlat.maskedImage.mask[:] = 0x0
    opticFlat.maskedImage.variance[:] = 0.0
    return opticFlat


def makePixelFlat(certifiedFlat, kernel='mean', window_size=40, mode='mirror', percentile=1., remove_background=True):
    logger = logging.getLogger()
    logger.info(f'Window size for {kernel} smoothing = {window_size}')
    pixelFlat = certifiedFlat.clone()
    opticFlat = makeOpticFlat(certifiedFlat, kernel=kernel, window_size=window_size, mode=mode, percentile=percentile)

    detector = opticFlat.getDetector()
    for amp in detector:
        bbox = amp.getBBox()
        pixelFlat[bbox].maskedImage.image.array[:, :] /= np.median(pixelFlat[bbox].maskedImage.image.array[:, :])
        pixelFlat[bbox].maskedImage.image.array[:, :] /= opticFlat[bbox].maskedImage.image.array[:, :]
        if remove_background:
            from astropy.stats import SigmaClip
            from photutils.background import Background2D, MedianBackground
            sigma_clip = SigmaClip(sigma=3.0)
            bkg_estimator = MedianBackground()
            bkg = Background2D(pixelFlat[bbox].maskedImage.image.array[:, :], (20, 20), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
            pixelFlat[bbox].maskedImage.image.array[:, :] /= bkg.background


            

    pixelFlat.maskedImage.mask[:] = 0x0
    pixelFlat.maskedImage.variance[:] = 0.0
    return pixelFlat


def makeSensorFlat(certifiedFlat, gainDict, kernel='mean', window_size=40, mode='mirror', percentile=1., invertGains=False, remove_background=True):
    logger = logging.getLogger()
    logger.info(f'Window size for {kernel} smoothing = {window_size}')
    pixelFlat = makePixelFlat(certifiedFlat, kernel=kernel, window_size=window_size, mode=mode,
                              percentile=percentile, remove_background=remove_background)
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
