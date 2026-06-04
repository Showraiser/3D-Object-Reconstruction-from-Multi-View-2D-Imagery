#!/bin/bash
mkdir -p datasets/ShapeNetCoreV2
cd datasets/ShapeNetCoreV2

# Adjust the paths to point to where your zip files are stored
for zip in ../../02691156.zip            ../../02747177.zip            ../../02773838.zip            ../../02801938.zip            ../../02808440.zip            ../../02818832.zip            ../../02828884.zip            ../../02834778.zip            ../../02843684.zip            ../../02858304.zip            ../../02871439.zip            ../../02876657.zip            ../../02880940.zip
do
  echo "Unzipping $zip ..."
  unzip -q $zip -d .
done
