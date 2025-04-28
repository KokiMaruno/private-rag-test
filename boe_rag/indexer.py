#!/usr/bin/env python
"""
テキストデータを読み込み、ベクトル化し、DuckDBに保存するモジュール
"""
import os
import json
import glob
from typing import List, Dict, Any, Optional, Tuple
import duckdb
import logging
from pathlib import Path
from .embedder import get_embedder, PlamoEmbedder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentIndexer:
    """ドキュメントのインデクサー"""

    def __init__(
        self,
        db_path: str = "data/vector_db.duckdb",
        embedder: Optional[PlamoEmbedder] = None,
        vector_dim: int = 2048,
        table_name: str = "documents"
    ):
        """
        Args:
            db_path: DuckDBのデータベースパス
            embedder: ベクトル化用のエンベッダー（Noneの場合は自動作成）
            vector_dim: ベクトルの次元数
            table_name: ベクトルを保存するテーブル名
        """
        self.db_path = db_path
        self.table_name = table_name
        self.vector_dim = vector_dim

        # DuckDBの接続を作成
        self._create_db_connection()

        # エンベッダーの設定
        self.embedder = embedder if embedder is not None else get_embedder()

    def _create_db_connection(self):
        """DuckDBの接続を作成し、VSSを設定する"""
        # データベースの親ディレクトリが存在することを確認
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        try:
            # DuckDBに接続
            self.conn = duckdb.connect(self.db_path)

            # VSS拡張をインストール・ロード
            self.conn.execute("INSTALL vss")
            self.conn.execute("LOAD vss")

            # 利用可能な関数を表示（デバッグ用）
            try:
                functions = self.conn.execute(
                    "SELECT function_name FROM duckdb_functions() WHERE function_name LIKE '%similarity%'"
                ).fetchall()
                function_names = [f[0] for f in functions]
                logger.info(f"利用可能な類似度関数: {function_names}")
            except:
                logger.warning("類似度関数のリストを取得できませんでした")

            logger.info(f"DuckDBへの接続とVSS拡張のロードが完了しました: {self.db_path}")

        except Exception as e:
            logger.error(f"DuckDBの設定に失敗しました: {e}")
            raise

    def create_vector_table(self, drop_existing: bool = False):
        """ベクトルテーブルを作成する

        Args:
            drop_existing: 既存のテーブルを削除するかどうか
        """
        try:
            # 既存のテーブルを削除（オプション）
            if drop_existing:
                self.conn.execute(f"DROP TABLE IF EXISTS {self.table_name}")

            # シーケンスの作成（存在していなければ）
            self.conn.execute(
                "CREATE SEQUENCE IF NOT EXISTS id_sequence START 1")

            # テーブルの作成
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER DEFAULT nextval('id_sequence'),
                    title TEXT,
                    category TEXT,
                    content TEXT,
                    filename TEXT,
                    vector FLOAT[{self.vector_dim}]
                )
            """)

            logger.info(f"ベクトルテーブル'{self.table_name}'の作成が完了しました")

        except Exception as e:
            logger.error(f"ベクトルテーブルの作成に失敗しました: {e}")
            raise

    def load_documents_from_directory(self, directory: str) -> List[Dict[str, Any]]:
        """ディレクトリからドキュメントをロードする

        Args:
            directory: ドキュメントが格納されているディレクトリパス

        Returns:
            ドキュメントのリスト（辞書形式）
        """
        documents = []

        try:
            # メタデータがあれば読み込む
            metadata_path = os.path.join(directory, "metadata.json")
            metadata_dict = {}

            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    # ファイル名をキーとするメタデータ辞書を作成
                    metadata_dict = {item["filename"]
                        : item for item in metadata}

            # ディレクトリ内のテキストファイルを探索
            text_files = glob.glob(os.path.join(directory, "*.txt"))

            for file_path in text_files:
                filename = os.path.basename(file_path)

                # ファイルからコンテンツを読み込む
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # ドキュメント情報を作成
                doc = {
                    "filename": filename,
                    "content": content,
                    "title": "",
                    "category": ""
                }

                # メタデータがあれば追加
                if filename in metadata_dict:
                    doc["title"] = metadata_dict[filename]["title"]
                    doc["category"] = metadata_dict[filename]["category"]
                else:
                    # メタデータがない場合はコンテンツから抽出を試みる
                    lines = content.split("\n")
                    for line in lines:
                        if line.startswith("タイトル:"):
                            doc["title"] = line.replace("タイトル:", "").strip()
                        elif line.startswith("カテゴリ:"):
                            doc["category"] = line.replace("カテゴリ:", "").strip()

                documents.append(doc)

            logger.info(f"{len(documents)}件のドキュメントを読み込みました")
            return documents

        except Exception as e:
            logger.error(f"ドキュメントの読み込みに失敗しました: {e}")
            return []

    def index_documents(self, documents: List[Dict[str, Any]]):
        """ドキュメントをベクトル化してデータベースに保存する

        Args:
            documents: ドキュメントのリスト
        """
        try:
            # ドキュメントの内容を抽出
            contents = [doc["content"] for doc in documents]

            # ドキュメントをベクトル化
            logger.info(f"{len(contents)}件のドキュメントをベクトル化します...")
            vectors = self.embedder.embed_documents(contents)

            # ベクトルとドキュメントをデータベースに保存
            for doc, vector in zip(documents, vectors):
                self.conn.execute(
                    f"INSERT INTO {
                        self.table_name} (title, category, content, filename, vector) VALUES (?, ?, ?, ?, ?)",
                    [doc["title"], doc["category"],
                        doc["content"], doc["filename"], vector]
                )

            logger.info(f"{len(documents)}件のドキュメントをデータベースに保存しました")

        except Exception as e:
            logger.error(f"ドキュメントのインデックス化に失敗しました: {e}")
            raise

    def index_directory(self, directory: str, drop_existing: bool = False):
        """ディレクトリ内のドキュメントをインデックス化する

        Args:
            directory: ドキュメントが格納されているディレクトリパス
            drop_existing: 既存のテーブルを削除するかどうか
        """
        # ベクトルテーブルを作成
        self.create_vector_table(drop_existing=drop_existing)

        # ドキュメントを読み込む
        documents = self.load_documents_from_directory(directory)

        if documents:
            # ドキュメントをインデックス化
            self.index_documents(documents)
            return True

        return False

    def close(self):
        """データベース接続を閉じる"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logger.info("データベース接続を閉じました")


def create_index_from_directory(
    data_dir: str,
    db_path: str = "data/vector_db.duckdb",
    drop_existing: bool = False
) -> bool:
    """ディレクトリからインデックスを作成するヘルパー関数

    Args:
        data_dir: データディレクトリ
        db_path: データベースパス
        drop_existing: 既存のテーブルを削除するかどうか

    Returns:
        成功したかどうか
    """
    try:
        # インデクサーを作成
        indexer = DocumentIndexer(db_path=db_path)

        # ディレクトリをインデックス化
        success = indexer.index_directory(
            data_dir, drop_existing=drop_existing)

        # インデクサーを閉じる
        indexer.close()

        return success

    except Exception as e:
        logger.error(f"インデックス作成に失敗しました: {e}")
        return False


if __name__ == "__main__":
    # 動作確認用コード
    data_dir = "data/boe_mythology"
    db_path = "data/vector_db.duckdb"

    print(f"ディレクトリ {data_dir} からインデックスを作成します...")
    success = create_index_from_directory(
        data_dir=data_dir,
        db_path=db_path,
        drop_existing=True
    )

    if success:
        print("インデックス作成が完了しました！")
    else:
        print("インデックス作成に失敗しました。")
