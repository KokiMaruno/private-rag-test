# 意味論的検索システムサンプル

このプロジェクトは、意味論的検索システムの基本実装です。
Plamo-Embedding-1Bを使用してテキストをベクトル化し、DuckDB+VSSを使用してベクトル検索を行います。

## 機能

- ドキュメントのベクトル化と保存
- ベクトル検索による関連ドキュメントの検索
- シンプルな対話型インターフェース

## 必要環境

- Python 3.11以上
- Poetry（パッケージ管理に使用）

## インストール方法

### 1. 依存パッケージのインストール

```bash
poetry install
```

## 使用方法

### 1. サンプルデータの生成
- サンプルデータは架空の惑星「ボー」の神話に関する断片的な情報になります

```bash
poetry run python main.py generate-data
```

### 2. インデックスの作成

```bash
poetry run python main.py index
```

### 3. 検索の実行

```bash
# 単一クエリの検索
poetry run python main.py search --query "サン・ジュカールとは誰ですか？"

# 対話モード
poetry run python main.py interactive
```

## 設定オプション

### インデックス作成

```bash
poetry run python main.py index \
  --data-dir data/boe_mythology \  # データディレクトリ
  --db-path data/vector_db.duckdb \ # データベースパス
  --force                          # 既存のインデックスを上書き
```

### 検索

```bash
poetry run python main.py search \
  --query "質問文" \              # 検索クエリ
  --db-path data/vector_db.duckdb \ # データベースパス
  --top-k 3 \                     # 上位何件を返すか
  --threshold 0.5                 # 類似度のしきい値
```

### 対話モード

```bash
poetry run python main.py interactive \
  --db-path data/vector_db.duckdb \ # データベースパス
  --top-k 3 \                     # 上位何件を返すか
  --threshold 0.5                 # 類似度のしきい値
```

## プロジェクト構成

```
boe-mythology-rag/
├── poetry.lock              # Poetryのロックファイル
├── pyproject.toml          # Poetryの設定ファイル
├── main.py                 # メインスクリプト
├── boe_rag/                # プロジェクトソースコード
│   ├── __init__.py
│   ├── embedder.py         # ベクトル化モジュール
│   ├── indexer.py          # インデックス作成モジュール
│   ├── retriever.py        # 検索モジュール
│   └── utils.py            # ユーティリティ関数
├── data/                   # データディレクトリ
│   ├── boe_mythology/      # 惑星ボー神話データ
│   └── vector_db.duckdb    # ベクトルデータベース
└── README.md               # このファイル
```

## 拡張方法

### 異なるデータソースの使用

`data/`ディレクトリに新しいデータフォルダを作成し、テキストファイルを配置してください。
その後、以下のコマンドでインデックスを作成します。

```bash
poetry run python main.py index --data-dir data/your_data_dir
```

### LLMによる回答生成の追加

実装予定の機能です。検索結果をLLMに渡して、より自然な回答を生成する機能を追加する予定です。

## 技術的詳細

- **エンベディングモデル**: Plamo-Embedding-1B (Preferred Networks)
- **ベクトルデータベース**: DuckDB + VSS拡張
- **エンベディングの次元数**: 2048

## 参考資料

- [Slug-Quick: Custom RAG](https://voluntas.ghost.io/slug-quick-custom-rag/)
- [Plamo-Embedding-1B](https://huggingface.co/pfnet/plamo-embedding-1b)

# 備考 
- ほとんどのコードはClaude 3.7 Sonnetにより実装
