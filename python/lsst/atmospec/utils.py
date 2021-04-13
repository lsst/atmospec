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
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.log as lsstLog
import lsst.afw.geom as afwGeom
import lsst.geom as geom
from lsst.afw.cameraGeom import PIXELS, FOCAL_PLANE
from lsst.obs.lsst.translators.lsst import FILTER_DELIMITER

import astropy
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Distance
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier


AUXTEL_LOCATION = EarthLocation(lat=-30.244639*u.deg, lon=-70.749417*u.deg, height=2663*u.m)


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
    logger : `lsst.log.Log`
        Logger for logging warnings

    Returns
    -------
    rotatedExp : `lsst.afw.image.exposure.Exposure`
        A copy of the input exposure, rotated by nDegrees
    """
    nDegrees += 180  # rotations of zero give a 180 degree rotation for some reason
    nDegrees = nDegrees % 360

    if not logger:
        logger = lsstLog.getLogger('atmospec.utils')

    wcs = exp.getWcs()
    if not wcs:
        logger.warn("Can't rotate exposure without a wcs - returning exp unrotated")
        return exp.clone()  # return a clone so it's always returning a copy as this is what default does

    warper = afwMath.Warper(kernelName)
    if isinstance(exp, afwImage.ExposureU):
        # TODO: remove once this bug is fixed - DM-20258
        logger.info('Converting ExposureU to ExposureF due to bug')
        logger.info('Remove this workaround after DM-20258')
        exp = afwImage.ExposureF(exp, deep=True)

    detector = exp.getDetector()
    pixelScale = wcs.getPixelScale().asDegrees()
    crval = wcs.getSkyOrigin()
    rotAngle = geom.Angle(nDegrees, geom.degrees)
    cd = (geom.LinearTransform.makeScaling(pixelScale)
          * geom.LinearTransform.makeRotation(rotAngle))
    crpix = detector.transform(geom.Point2D(0, 0), FOCAL_PLANE, PIXELS)
    rotatedWcs = afwGeom.makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cd.getMatrix())

    rotatedExp = warper.warpExposure(rotatedWcs, exp)
    rotatedExp.setXY0(geom.Point2I(0, 0))
    return rotatedExp


def airMassFromRawMetadata(md):
    """Calculate the visit's airmass from the raw header information.

    Returns the airmass, or 0 if the calculation fails.
    Zero was chosen as it is an obviously unphysical value, but means
    that calling code doesn't have to test if None, as numeric values can
    be used more easily in place."""
    time = Time(md['DATE-OBS'])
    if md['RASTART'] is not None and md['DECSTART'] is not None:
        skyLocation = SkyCoord(md['RASTART'], md['DECSTART'], unit=u.deg)
    elif md['RA'] is not None and md['DEC'] is not None:
        skyLocation = SkyCoord(md['RA'], md['DEC'], unit=u.deg)
    else:
        return 0
    altAz = AltAz(obstime=time, location=AUXTEL_LOCATION)
    observationAltAz = skyLocation.transform_to(altAz)
    return observationAltAz.secz.value


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
        logger = lsstLog.Log.getDefaultLogger()

    resultFrom = None
    targetLocation = None
    # try vizier, but it is slow, unreliable, and
    # many objects are found but have no Hipparcos entries
    try:
        targetLocation = vizierLocationForTarget(exp, target, doMotionCorrection=doMotionCorrection)
        resultFrom = 'vizier'
        logger.info(f"Target location for {target} retrieved from Vizier")

    # fail over to simbad - it has ~every target, but no proper motions
    except ValueError:
        try:
            logger.warn(f"Target {target} not found in Vizier, failing over to try Simbad")
            targetLocation = simbadLocationForTarget(target)
            resultFrom = 'simbad'
            logger.info(f"Target location for {target} retrieved from Simbad")
        except ValueError as inst:  # simbad found zero or several results for target
            msg = inst.args[0]
            logger.warn(msg)
            return None

    if not targetLocation:
        return None

    if doMotionCorrection and resultFrom == 'simbad':
        logger.warn(f"Failed to apply motion correction because {target} was"
                    " only found in Simbad, not Vizier/Hipparcos")

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
    obj = Simbad.query_object(target)
    if not obj:
        raise ValueError(f"Found failed to find {target} in simbad!")
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
    """Check if an exposure is dispersed."""
    filterFullName = exp.getFilterLabel().physicalLabel
    if FILTER_DELIMITER not in filterFullName:
        raise RuntimeError(f"Error parsing filter name {filterFullName}")
    filt, grating = filterFullName.split(FILTER_DELIMITER)
    if grating.upper().startswith('EMPTY'):
        return False
    return True


def isDispersedDataId(dataId, butler):
    """Check if a dataId corresponds to a dispersed image."""
    filterFullName = butler.queryMetadata('raw', 'filter', **dataId)[0]
    if FILTER_DELIMITER not in filterFullName:
        raise RuntimeError(f"Error parsing filter name {filterFullName}")
    filt, grating = filterFullName.split(FILTER_DELIMITER)
    if grating.upper().startswith('EMPTY'):
        return False
    return True
