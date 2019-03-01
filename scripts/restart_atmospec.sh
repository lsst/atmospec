AUXTEL_REPO_DIR=~/atmospec_dev
AUXTEL_DATA_DIR=~/atmospec_testdata_1

whoami=$(whoami)
batchArgs="--batch-type none"

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

rm -rf $AUXTEL_REPO_DIR
mkdir -p $AUXTEL_REPO_DIR

echo "lsst.obs.lsst.auxTel.AuxTelMapper" > $AUXTEL_REPO_DIR/_mapper

ingestImages.py $AUXTEL_REPO_DIR $AUXTEL_DATA_DIR/*.fits

constructBias.py $AUXTEL_REPO_DIR --rerun calibs --id dayObs='2018-09-20' seqNum=28..37 --batch-type none
ingestCalibs.py $AUXTEL_REPO_DIR --calibType bias $AUXTEL_REPO_DIR/rerun/calibs/bias/*/*.fits --validity 9999 --output $AUXTEL_REPO_DIR/CALIB --mode=link

python makeFakeFlats.py

ingestCalibs.py $AUXTEL_REPO_DIR --calibType flat $AUXTEL_REPO_DIR/CALIB/bias/*/*fakeFlat*.fits --validity 9999 --output $AUXTEL_REPO_DIR/CALIB --mode=link
