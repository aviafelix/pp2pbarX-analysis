#!/bin/sh
# nohup google-chrome --incognito ./*.html html/*.html > /dev/null 2>&1 &
google-chrome --incognito ./*.html html/*.html > /dev/null 2>&1 &
disown
