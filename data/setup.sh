#!/bin/bash
# 惑星ボー神話RAGシステムのセットアップスクリプト

# エラーが発生したら終了
set -e

# 現在のディレクトリに移動（スクリプトがどこから実行されても機能するように）
cd "$(dirname "$0")"

echo "===== 惑星ボー神話RAGシステムのセットアップを開始します ====="

# Poetryが存在するか確認
if ! command -v poetry &> /dev/null; then
    echo "Poetryがインストールされていません。インストール方法:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

echo "1. 依存パッケージをインストールします..."
poetry install

echo "2. サンプルデータを生成します..."
poetry run python main.py generate-data

echo "3. インデックスを作成します..."
poetry run python main.py index --force

echo "===== セットアップが完了しました ====="
echo "以下のコマンドで対話モードを開始できます:"
echo "poetry run python main.py interactive"
