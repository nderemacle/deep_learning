#!/usr/bin/env bash

set -e

pip install Sphinx
pip install rinohtype
pip install sphinx_rtd_theme
pip install sphinx-autodoc-typehints
pip install sphinxcontrib-napoleon

cd doc
make html