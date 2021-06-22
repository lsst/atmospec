import lsst.daf.persistence as dafPersist
from lsst.atmospec.utils import makeGainFlat
from astropy.io import fits

# gainDict = {'C00': 1,
#             'C01': 1,
#             'C02': 1,
#             'C03': 1,
#             'C04': 1,
#             'C05': 1,
#             'C06': 1,
#             'C07': 1,
#             'C10': 1,
#             'C11': 1,
#             'C12': 1,
#             'C13': 1,
#             'C14': 1,
#             'C15': 1,
#             'C16': 1,
#             'C17': 1}

# data from
# https://lsst-camera.slac.stanford.edu/DataPortal/RawReport.jsp?run=74
gainDict = {'C00': 2.4837722778320312,
            'C01': 2.4612927436828613,
            'C02': 2.4486501216888428,
            'C03': 2.457714796066284,
            'C04': 2.4735212326049805,
            'C05': 2.479198694229126,
            'C06': 2.5018515586853027,
            'C07': 2.487159013748169,
            'C10': 2.494577169418335,
            'C11': 2.4739749431610107,
            'C12': 2.4664266109466553,
            'C13': 2.473435163497925,
            'C14': 2.4772729873657227,
            'C15': 2.472003221511841,
            'C16': 2.443410873413086,
            'C17': 2.4504425525665283}

butler = dafPersist.Butler('/home/mfl/atmospec_dev/')
dataId = {'dayObs': '2018-09-20', 'seqNum': 60}
bias = butler.get('bias', dataId=dataId)

biasFilename = butler.get('bias_filename', dataId=dataId)[0]
fakeFlat = makeGainFlat(bias, gainDict)

filters = ['u', 'g', 'r', 'i', 'z', 'y', 'NONE']

for filt in filters:
    fakeFlatFilename = biasFilename[:-5] + '_fakeFlat_{}.fits'.format(filt)
    fakeFlat.writeFits(fakeFlatFilename)

    fits.setval(fakeFlatFilename, 'OBSTYPE', value='flat')
    fits.setval(fakeFlatFilename, 'FILTER', value=filt)

    newCalibId = []
    for item in fits.getval(fakeFlatFilename, 'CALIB_ID').split(' '):
        if not item.startswith('filter'):
            newCalibId.append(item)
        else:
            newCalibId.append('filter={}'.format(filt))
    newCalibId = ' '.join(item for item in newCalibId)

    fits.setval(fakeFlatFilename, 'CALIB_ID', value=newCalibId)
    print('Wrote fake flat for {} to {}'.format(filt, fakeFlatFilename))
