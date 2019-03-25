#!/usr/bin/env bash

set -e

pip install Sphinx
pip install rinohtype
pip install sphinx_rtd_theme
pip install sphinx-autodoc-typehints
pip install sphinxcontrib-napoleon

mkdir doc
cd doc

sphinx-quickstart --quiet \
                  --sep \
                  --project='deep_learning' \
                  --author='Nicolas de Remacle' \
                  --v '0.1.0' \
                  --language='en' \
                  --no-batchfile \


