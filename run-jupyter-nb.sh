#!/bin/sh
jupyter-notebook --no-browser --ip=$(resolveip -s $(hostname)) --port=9731 > /dev/null 2>&1 &
# html+ipynb