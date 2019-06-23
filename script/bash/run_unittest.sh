#!/usr/bin/env bash

export PYTHONPATH=$(pwd)

export PATH_FOLDER=test/deep_learning

for file in test_base_operator test_layer test_loss test_tf_utils
do
    python ${PATH_FOLDER}/${file}.py
done

