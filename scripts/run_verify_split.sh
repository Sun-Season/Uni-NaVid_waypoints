#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate uninavid

cd /mnt/dataset/shuhzeng/Uni-NaVid_waypoints
python scripts/verify_split.py
