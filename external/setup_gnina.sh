#!/bin/bash

export GITHUB_BINARY_DIR=external/gnina

wget -P $GITHUB_BINARY_DIR https://github.com/gnina/gnina/releases/download/v1.1/gnina
chmod +x "$GITHUB_BINARY_DIR/gnina"
