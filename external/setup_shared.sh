#!/bin/bash

export GITHUB_NAME="$GITHUB_ORG_NAME/$GITHUB_REPO_NAME"
export MODEL_DIR="external/$GITHUB_REPO_DIR"

echo "Setting up $GITHUB_NAME under $MODEL_DIR"

git -C external clone "https://github.com/$GITHUB_NAME" $GITHUB_REPO_DIR
git -C $MODEL_DIR checkout $GITHUB_COMMIT_ID
