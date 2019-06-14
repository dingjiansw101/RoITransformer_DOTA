#!/bin/bash
set - x
set - e
export PYTHONUNBUFFERED="True"

python ImgSplit.py --dataset /data/dota_new/dota/val --dest /data/dota_new/dota/split/val --scale 1.0

