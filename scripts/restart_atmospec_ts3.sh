AUXTEL_REPO_DIR=~/atmospec_ts3
AUXTEL_DATA_DIR=/home/mfl/atmospec_ts3_data

whoami=$(whoami)
batchArgs="--batch-type none"

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

rm -rf $AUXTEL_REPO_DIR
mkdir -p $AUXTEL_REPO_DIR

echo "lsst.obs.lsst.auxTel.AuxTelMapper" > $AUXTEL_REPO_DIR/_mapper

ingestImages.py $AUXTEL_REPO_DIR $AUXTEL_DATA_DIR/*/v0/278/*.fits
