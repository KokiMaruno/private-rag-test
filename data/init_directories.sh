#!/bin/bash
# プロジェクトディレクトリ構造を作成するスクリプト

# エラーが発生したら終了
set -e

# 現在のディレクトリに移動（スクリプトがどこから実行されても機能するように）
cd "$(dirname "$0")"

# 必要なディレクトリを作成
mkdir -p boe_rag
mkdir -p data/boe_mythology
mkdir -p tests
mkdir -p scripts

# 必要な空ファイルを作成
touch boe_rag/__init__.py
touch tests/__init__.py
touch scripts/__init__.py

# スクリプトディレクトリに実行権限を付与
chmod -R +x scripts/

echo "ディレクトリ構造の作成が完了しました。"
