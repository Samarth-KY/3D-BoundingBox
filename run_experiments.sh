#!/bin/bash

python train.py --run_name "instance-only" --num_epochs 200
python train.py --scene_context --run_name "with-scene-context" --num_epochs 200
python train.py --scene_context --augment --run_name "with-scene-context-and-aug" --num_epochs 200