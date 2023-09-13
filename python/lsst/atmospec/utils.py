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

__all__ = [
    "argMaxNd",
    "gainFromFlatPair",
    "getAmpReadNoiseFromRawExp",
    "getLinearStagePosition",
    "getFilterAndDisperserFromExp",
    "getSamplePoints",
    "getTargetCentroidFromWcs",
    "isDispersedDataId",
    "isDispersedExp",
    "isExposureTrimmed",
    "makeGainFlat",
    "rotateExposure",
    "simbadLocationForTarget",
    "vizierLocationForTarget",
    "runNotebook",
]

import logging
import numpy as np
import sys
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
from lsst.ctrl.mpexec import SimplePipelineExecutor
import lsst.afw.geom as afwGeom
import lsst.geom as geom
import lsst.daf.butler as dafButler
from astro_metadata_translator import ObservationInfo
import lsst.pex.config as pexConfig
from lsst.pipe.base import Pipeline
from lsst.obs.lsst.translators.lsst import FILTER_DELIMITER
from lsst.utils.iteration import ensure_iterable

import astropy
import astropy.units as u
from astropy.coordinates import SkyCoord, Distance


def makeGainFlat(exposure, gainDict, invertGains=False):
    """Given an exposure, make a flat from the gains.

    Construct an exposure where the image array data contains the gain of the
    pixel, such that dividing by (or mutiplying by) the image will convert
    an image from ADU to electrons.

    Parameters
    ----------
    detectorExposure : `lsst.afw.image.exposure`
        The template exposure for which the flat is to be made.

    gainDict : `dict` of `float`
        A dict of the amplifiers' gains, keyed by the amp names.

    invertGains : `bool`
        Gains are specified in inverted units and should be applied as such.

    Returns
    -------
    gainFlat : `lsst.afw.image.exposure`
        The gain flat
    """
    flat = exposure.clone()
    detector = flat.getDetector()
    ampNames = set(list(a.getName() for a in detector))
    assert set(gainDict.keys()) == ampNames

    for amp in detector:
        bbox = amp.getBBox()
        if invertGains:
            flat[bbox].maskedImage.image.array[:, :] = 1./gainDict[amp.getName()]
        else:
            flat[bbox].maskedImage.image.array[:, :] = gainDict[amp.getName()]
    flat.maskedImage.mask[:] = 0x0
    flat.maskedImage.variance[:] = 0.0

    return flat


def argMaxNd(array):
    """Get the index of the max value of an array.

    If there are multiple occurences of the maximum value
    just return the first.
    """
    return np.unravel_index(np.argmax(array, axis=None), array.shape)


def getSamplePoints(start, stop, nSamples, includeEndpoints=False, integers=False):
    """Get the locations of the coordinates to use to sample a range evenly

    Divide a range up and return the coordinated to use in order to evenly
    sample the range. If asking for integers, rounded values are returned,
    rather than int-truncated ones.

    If not including the endpoints, divide the (stop-start) range into nSamples
    and return the midpoint of each section, thus leaving a sectionLength/2 gap
    between the first/last samples and the range start/end.

    If including the endpoints, the first and last points will be
    start, stop respectively, and other points will be the endpoints of the
    remaining nSamples-1 sections.

    Visually, for a range:

    |--*--|--*--|--*--|--*--| return * if not including end points, n=4
    |-*-|-*-|-*-|-*-|-*-|-*-| return * if not including end points, n=6

    *-----*-----*-----*-----* return * if we ARE including end points, n=4
    *---*---*---*---*---*---* return * if we ARE including end points, n=6
    """

    if not includeEndpoints:
        r = (stop-start)/(2*nSamples)
        points = [((2*pointNum+1)*r) for pointNum in range(nSamples)]
    else:
        if nSamples <= 1:
            raise RuntimeError('nSamples must be >= 2 if including endpoints')
        if nSamples == 2:
            points = [start, stop]
        else:
            r = (stop-start)/(nSamples-1)
            points = [start]
            points.extend([((pointNum)*r) for pointNum in range(1, nSamples)])

    if integers:
        return [int(x) for x in np.round(points)]
    return points


def isExposureTrimmed(exp):
    det = exp.getDetector()
    if exp.getDimensions() == det.getBBox().getDimensions():
        return True
    return False


def getAmpReadNoiseFromRawExp(rawExp, ampNum, nOscanBorderPix=0):
    """XXX doctring here

    Trim identically in all direction for convenience"""
    if isExposureTrimmed(rawExp):
        raise RuntimeError('Got an assembled exposure instead of a raw one')

    det = rawExp.getDetector()

    amp = det[ampNum]
    if nOscanBorderPix == 0:
        noise = np.std(rawExp[amp.getRawHorizontalOverscanBBox()].image.array)
    else:
        b = nOscanBorderPix  # line length limits :/
        noise = np.std(rawExp[amp.getRawHorizontalOverscanBBox()].image.array[b:-b, b:-b])
    return noise


def gainFromFlatPair(flat1, flat2, correctionType=None, rawExpForNoiseCalc=None, overscanBorderSize=0):
    """Calculate the gain from a pair of flats.

    The basic premise is 1/g = <(I1 - I2)^2/(I1 + I2)>
    Corrections for the variable QE and the read-noise are then made
    following the derivation in Robert's forthcoming book, which gets

    1/g = <(I1 - I2)^2/(I1 + I2)> - 1/mu(sigma^2 - 1/2g^2)

    If you are lazy, see below for the solution.
    https://www.wolframalpha.com/input/?i=solve+1%2Fy+%3D+c+-+(1%2Fm)*(s^2+-+1%2F(2y^2))+for+y

    where mu is the average signal level, and sigma is the
    amplifier's readnoise. The way the correction is applied depends on
    the value supplied for correctionType.

    correctionType is one of [None, 'simple' or 'full']
        None     : uses the 1/g = <(I1 - I2)^2/(I1 + I2)> formula
        'simple' : uses the gain from the None method for the 1/2g^2 term
        'full'   : solves the full equation for g, discarding the non-physical
                   solution to the resulting quadratic

    Parameters
    ----------
    flat1 : `lsst.afw.image.exposure`
        The first of the postISR assembled, overscan-subtracted flat pairs

    flat2 : `lsst.afw.image.exposure`
        The second of the postISR assembled, overscan-subtracted flat pairs

    correctionType : `str` or `None`
        The correction applied, one of [None, 'simple', 'full']

    rawExpForNoiseCalc : `lsst.afw.image.exposure`
        A raw (un-assembled) image from which to measure the noise

    overscanBorderSize : `int`
        The number of pixels to crop from the overscan region in all directions

    Returns
    -------
    gainDict : `dict`
        Dictionary of the amplifier gains, keyed by ampName
    """
    if correctionType not in [None, 'simple', 'full']:
        raise RuntimeError("Unknown correction type %s" % correctionType)

    if correctionType is not None and rawExpForNoiseCalc is None:
        raise RuntimeError("Must supply rawFlat if performing correction")

    gains = {}
    det = flat1.getDetector()
    for ampNum, amp in enumerate(det):
        i1 = flat1[amp.getBBox()].image.array
        i2 = flat2[amp.getBBox()].image.array
        const = np.mean((i1 - i2)**2 / (i1 + i2))
        basicGain = 1. / const

        if correctionType is None:
            gains[amp.getName()] = basicGain
            continue

        mu = (np.mean(i1) + np.mean(i2)) / 2.
        sigma = getAmpReadNoiseFromRawExp(rawExpForNoiseCalc, ampNum, overscanBorderSize)

        if correctionType == 'simple':
            simpleGain = 1/(const - (1/mu)*(sigma**2 - (1/2*basicGain**2)))
            gains[amp.getName()] = simpleGain

        elif correctionType == 'full':
            root = np.sqrt(mu**2 - 2*mu*const + 2*sigma**2)
            denom = (2*const*mu - 2*sigma**2)

            positiveSolution = (root + mu)/denom
            negativeSolution = (mu - root)/denom  # noqa: F841 unused, but the other solution

            gains[amp.getName()] = positiveSolution

    return gains


def rotateExposure(exp, nDegrees, kernelName='lanczos4', logger=None):
    """Rotate an exposure by nDegrees clockwise.

    Parameters
    ----------
    exp : `lsst.afw.image.exposure.Exposure`
        The exposure to rotate
    nDegrees : `float`
        Number of degrees clockwise to rotate by
    kernelName : `str`
        Name of the warping kernel, used to instantiate the warper.
    logger : `logging.Logger`
        Logger for logging warnings

    Returns
    -------
    rotatedExp : `lsst.afw.image.exposure.Exposure`
        A copy of the input exposure, rotated by nDegrees
    """
    nDegrees = nDegrees % 360

    if not logger:
        logger = logging.getLogger(__name__)

    wcs = exp.getWcs()
    if not wcs:
        logger.warning("Can't rotate exposure without a wcs - returning exp unrotated")
        return exp.clone()  # return a clone so it's always returning a copy as this is what default does

    warper = afwMath.Warper(kernelName)
    if isinstance(exp, afwImage.ExposureU):
        # TODO: remove once this bug is fixed - DM-20258
        logger.info('Converting ExposureU to ExposureF due to bug')
        logger.info('Remove this workaround after DM-20258')
        exp = afwImage.ExposureF(exp, deep=True)

    affineRotTransform = geom.AffineTransform.makeRotation(nDegrees*geom.degrees)
    transformP2toP2 = afwGeom.makeTransform(affineRotTransform)
    rotatedWcs = afwGeom.makeModifiedWcs(transformP2toP2, wcs, False)

    rotatedExp = warper.warpExposure(rotatedWcs, exp)
    # rotatedExp.setXY0(geom.Point2I(0, 0))  # TODO: check no longer required
    return rotatedExp


def airMassFromRawMetadata(md):
    """Calculate the visit's airmass from the raw header information.

    Parameters
    ----------
    md : `Mapping`
        The raw header.

    Returns
    -------
    airmass : `float`
        Returns the airmass, or 0.0 if the calculation fails.
        Zero was chosen as it is an obviously unphysical value, but means
        that calling code doesn't have to test if None, as numeric values can
        be used more easily in place.
    """
    try:
        obsInfo = ObservationInfo(md, subset={"boresight_airmass"})
    except Exception:
        return 0.0
    return obsInfo.boresight_airmass


def getTargetCentroidFromWcs(exp, target, doMotionCorrection=True, logger=None):
    """Get the target's centroid, given an exposure with fitted WCS.

    Parameters
    ----------
    exp : `lsst.afw.exposure.Exposure`
        Exposure with fitted WCS.

    target : `str`
        The name of the target, e.g. 'HD 55852'

    doMotionCorrection : `bool`, optional
        Correct for proper motion and parallax if possible.
        This requires the object is found in Vizier rather than Simbad.
        If that is not possible, a warning is logged, and the uncorrected
        centroid is returned.

    Returns
    -------
    pixCoord : `tuple` of `float`, or `None`
        The pixel (x, y) of the target's centroid, or None if the object
        is not found.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    resultFrom = None
    targetLocation = None
    # try vizier, but it is slow, unreliable, and
    # many objects are found but have no Hipparcos entries
    try:
        targetLocation = vizierLocationForTarget(exp, target, doMotionCorrection=doMotionCorrection)
        resultFrom = 'vizier'
        logger.info("Target location for %s retrieved from Vizier", target)

    # fail over to simbad - it has ~every target, but no proper motions
    except ValueError:
        try:
            logger.warning("Target %s not found in Vizier, failing over to try Simbad", target)
            targetLocation = simbadLocationForTarget(target)
            resultFrom = 'simbad'
            logger.info("Target location for %s retrieved from Simbad", target)
        except ValueError as inst:  # simbad found zero or several results for target
            logger.warning("%s", inst.args[0])
            return None

    if not targetLocation:
        return None

    if doMotionCorrection and resultFrom == 'simbad':
        logger.warning("Failed to apply motion correction because %s was"
                       " only found in Simbad, not Vizier/Hipparcos", target)

    pixCoord = exp.getWcs().skyToPixel(targetLocation)
    return pixCoord


def simbadLocationForTarget(target):
    """Get the target location from Simbad.

    Parameters
    ----------
    target : `str`
        The name of the target, e.g. 'HD 55852'

    Returns
    -------
    targetLocation : `lsst.geom.SpherePoint`
        Nominal location of the target object, uncorrected for
        proper motion and parallax.

    Raises
    ------
    ValueError
        If object not found, or if multiple entries for the object are found.
    """
    # do not import at the module level - tests crash due to a race
    # condition with directory creation
    from astroquery.simbad import Simbad

    obj = Simbad.query_object(target)
    if not obj:
        raise ValueError(f"Failed to find {target} in simbad!")
    if len(obj) != 1:
        raise ValueError(f"Found {len(obj)} simbad entries for {target}!")

    raStr = obj[0]['RA']
    decStr = obj[0]['DEC']
    skyLocation = SkyCoord(raStr, decStr, unit=(u.hourangle, u.degree), frame='icrs')
    raRad, decRad = skyLocation.ra.rad, skyLocation.dec.rad
    ra = geom.Angle(raRad)
    dec = geom.Angle(decRad)
    targetLocation = geom.SpherePoint(ra, dec)
    return targetLocation


def vizierLocationForTarget(exp, target, doMotionCorrection):
    """Get the target location from Vizier optionally correction motion.

    Parameters
    ----------
    target : `str`
        The name of the target, e.g. 'HD 55852'

    Returns
    -------
    targetLocation : `lsst.geom.SpherePoint` or `None`
        Location of the target object, optionally corrected for
        proper motion and parallax.

    Raises
    ------
    ValueError
        If object not found in Hipparcos2 via Vizier.
        This is quite common, even for bright objects.
    """
    # do not import at the module level - tests crash due to a race
    # condition with directory creation
    from astroquery.vizier import Vizier

    result = Vizier.query_object(target)  # result is an empty table list for an unknown target
    try:
        star = result['I/311/hip2']
    except TypeError:  # if 'I/311/hip2' not in result (result doesn't allow easy checking without a try)
        raise ValueError

    epoch = "J1991.25"
    coord = SkyCoord(ra=star[0]['RArad']*u.Unit(star['RArad'].unit),
                     dec=star[0]['DErad']*u.Unit(star['DErad'].unit),
                     obstime=epoch,
                     pm_ra_cosdec=star[0]['pmRA']*u.Unit(star['pmRA'].unit),  # NB contains cosdec already
                     pm_dec=star[0]['pmDE']*u.Unit(star['pmDE'].unit),
                     distance=Distance(parallax=star[0]['Plx']*u.Unit(star['Plx'].unit)))

    if doMotionCorrection:
        expDate = exp.getInfo().getVisitInfo().getDate()
        obsTime = astropy.time.Time(expDate.get(expDate.EPOCH), format='jyear', scale='tai')
        newCoord = coord.apply_space_motion(new_obstime=obsTime)
    else:
        newCoord = coord

    raRad, decRad = newCoord.ra.rad, newCoord.dec.rad
    ra = geom.Angle(raRad)
    dec = geom.Angle(decRad)
    targetLocation = geom.SpherePoint(ra, dec)
    return targetLocation


def isDispersedExp(exp):
    """Check if an exposure is dispersed.

    Parameters
    ----------
    exp : `lsst.afw.image.Exposure`
        The exposure.

    Returns
    -------
    isDispersed : `bool`
        Whether it is a dispersed image or not.
    """
    filterFullName = exp.filter.physicalLabel
    if FILTER_DELIMITER not in filterFullName:
        raise RuntimeError(f"Error parsing filter name {filterFullName}")
    filt, grating = filterFullName.split(FILTER_DELIMITER)
    if grating.upper().startswith('EMPTY'):
        return False
    return True


def isDispersedDataId(dataId, butler):
    """Check if a dataId corresponds to a dispersed image.

    Parameters
    ----------
    dataId : `dict`
        The dataId.
    butler : `lsst.daf.butler.Butler`
        The butler.

    Returns
    -------
    isDispersed : `bool`
        Whether it is a dispersed image or not.
    """
    if isinstance(butler, dafButler.Butler):
        # TODO: DM-38265 Need to make this work with DataCoords
        assert 'day_obs' in dataId or 'exposure.day_obs' in dataId, f'failed to find day_obs in {dataId}'
        assert 'seq_num' in dataId or 'exposure.seq_num' in dataId, f'failed to find seq_num in {dataId}'
        seq_num = dataId['seq_num'] if 'seq_num' in dataId else dataId['exposure.seq_num']
        day_obs = dataId['day_obs'] if 'day_obs' in dataId else dataId['exposure.day_obs']
        where = "exposure.day_obs=day_obs AND exposure.seq_num=seq_num"
        expRecords = butler.registry.queryDimensionRecords("exposure", where=where,
                                                           bind={'day_obs': day_obs,
                                                                 'seq_num': seq_num})
        expRecords = set(expRecords)
        assert len(expRecords) == 1, f'Found more than one exposure record for {dataId}'
        filterFullName = expRecords.pop().physical_filter
    else:
        raise RuntimeError(f'Expected a butler, got {type(butler)}')
    if FILTER_DELIMITER not in filterFullName:
        raise RuntimeError(f"Error parsing filter name {filterFullName}")
    filt, grating = filterFullName.split(FILTER_DELIMITER)
    if grating.upper().startswith('EMPTY'):
        return False
    return True


def getLinearStagePosition(exp):
    """Get the linear stage position.

    Parameters
    ----------
    exp : `lsst.afw.image.Exposure`
        The exposure.

    Returns
    -------
    position : `float`
        The position of the linear stage, in mm.
    """
    md = exp.getMetadata()
    linearStagePosition = 115  # this seems to be the rough zero-point for some reason
    if 'LINSPOS' in md:
        position = md['LINSPOS']  # linear stage position in mm from CCD, larger->further from CCD
        if position is not None:
            linearStagePosition += position
    return linearStagePosition


def getFilterAndDisperserFromExp(exp):
    """Get the filter and disperser from an exposure.

    Parameters
    ----------
    exp : `lsst.afw.image.Exposure`
        The exposure.

    Returns
    -------
    filter, disperser : `tuple` of `str`
        The filter and the disperser names, as strings.
    """
    filterFullName = exp.getFilter().physicalLabel
    if FILTER_DELIMITER not in filterFullName:
        filt = filterFullName
        grating = exp.getInfo().getMetadata()['GRATING']
    else:
        filt, grating = filterFullName.split(FILTER_DELIMITER)
    return filt, grating


def runNotebook(dataId,
                outputCollection,
                *,
                extraInputCollections=None,
                taskConfigs={},
                configOptions={},
                embargo=False):
    """Run the ProcessStar pipeline for a single dataId, writing to the
    specified output collection.

    This is a convenience function to allow running single dataIds in notebooks
    so that plots can be inspected easily. This is not designed for bulk data
    reductions.

    Parameters
    ----------
    dataId : `dict`
        The dataId to run.
    outputCollection : `str`, optional
        Output collection name.
    extraInputCollections : `list` of `str`
        Any extra input collections to use when processing.
    taskConfigs : `dict` [`lsst.pipe.base.PipelineTaskConfig`], optional
        Dictionary of config config classes. The key of the ``taskConfigs``
        dict is the relevant task label. The value of ``taskConfigs``
        is a task config object to apply. See notes for ignored items.
    configOptions : `dict` [`dict`], optional
        Dictionary of individual config options. The key of the
        ``configOptions`` dict is the relevant task label. The value
        of ``configOptions`` is another dict that contains config
        key/value overrides to apply.
    embargo : `bool`, optional
        Use the embargo repo?

    Returns
    -------
    spectraction : `lsst.atmospec.spectraction.Spectraction`
        The extracted spectraction object.

    Notes
    -----
    Any ConfigurableInstances in supplied task config overrides will be
    ignored. Currently (see DM-XXXXX) this causes a RecursionError.
    """
    def makeQuery(dataId):
        dayObs = dataId['day_obs'] if 'day_obs' in dataId else dataId['exposure.day_obs']
        seqNum = dataId['seq_num'] if 'seq_num' in dataId else dataId['exposure.seq_num']
        queryString = (f"exposure.day_obs={dayObs} AND "
                       f"exposure.seq_num={seqNum} AND "
                       "instrument='LATISS'")

        return queryString
    repo = "LATISS" if not embargo else "/repo/embargo"

    # TODO: use LATISS_DEFAULT_COLLECTIONS here?
    inputs = ['LATISS/raw/all', 'refcats', 'LATISS/calib']
    if extraInputCollections is not None:
        extraInputCollections = ensure_iterable(extraInputCollections)
        inputs.extend(extraInputCollections)
    butler = SimplePipelineExecutor.prep_butler(repo,
                                                inputs=inputs,
                                                output=outputCollection)

    pipeline = Pipeline.fromFile("${ATMOSPEC_DIR}/pipelines/processStar.yaml")

    for taskName, configClass in taskConfigs.items():
        for option, value in configClass.items():
            # connections require special treatment
            if isinstance(value, configClass.ConnectionsConfigClass):
                for connectionOption, connectionValue in value.items():
                    pipeline.addConfigOverride(taskName,
                                               f'{option}.{connectionOption}',
                                               connectionValue)
            # ConfigurableInstance has to be done with .retarget()
            elif not isinstance(value, pexConfig.ConfigurableInstance):
                pipeline.addConfigOverride(taskName, option, value)

    for taskName, configDict in configOptions.items():
        for option, value in configDict.items():
            # ConfigurableInstance has to be done with .retarget()
            if not isinstance(value, pexConfig.ConfigurableInstance):
                pipeline.addConfigOverride(taskName, option, value)

    query = makeQuery(dataId)
    executor = SimplePipelineExecutor.from_pipeline(pipeline,
                                                    where=query,
                                                    root=repo,
                                                    butler=butler)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    quanta = executor.run()

    # quanta is just a plain list, but the items know their names, so rather
    # than just taking the third item and relying on that being the position in
    # the pipeline we get the item by name
    processStarQuantum = [q for q in quanta if q.taskName == 'lsst.atmospec.processStar.ProcessStarTask'][0]
    dataRef = processStarQuantum.outputs['spectractorSpectrum'][0]
    result = butler.get(dataRef)
    return result
