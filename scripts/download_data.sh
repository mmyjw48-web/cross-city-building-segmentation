#!/usr/bin/env bash
# Download the SpaceNet-2 training imagery + building footprints for Paris and
# Khartoum from the public S3 bucket (no AWS account needed).
#
#   bash scripts/download_data.sh [DEST]      # default DEST=data
#
# Size: ~2.8 GB (Paris) + ~2.5 GB (Khartoum). Requires the AWS CLI (pip install awscli).
set -euo pipefail

DEST="${1:-data}"
BUCKET="s3://spacenet-dataset/spacenet/SN2_buildings/train"

for AOI in AOI_3_Paris AOI_5_Khartoum; do
  for SUB in PS-RGB geojson_buildings; do
    mkdir -p "$DEST/$AOI/$SUB"
    aws s3 sync --no-sign-request --only-show-errors \
      "$BUCKET/$AOI/$SUB/" "$DEST/$AOI/$SUB/"
  done
done

for AOI in AOI_3_Paris AOI_5_Khartoum; do
  echo "$AOI: $(ls "$DEST/$AOI/PS-RGB" | wc -l) images, $(ls "$DEST/$AOI/geojson_buildings" | wc -l) geojson"
done
# Expected: AOI_3_Paris 1148 / 1148, AOI_5_Khartoum 1012 / 1012
