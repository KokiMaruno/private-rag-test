#!/usr/bin/env python
"""
ベクトル検索を行うモジュール
ユーザーの質問をベクトル化し、似たドキュメントを検索する
"""
import os
import duckdb
import logging
from typing import List, Dict, Any, Optional, Tuple
from .embedder import get_embedder, PlamoEmbedder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorRetriever:
    """ベクトル検索を行うクラス"""

    def __init__(
        self,
        db_path: str = "data/vector_db.duckdb",
        embedder: Optional[PlamoEmbedder] = None,
        table_name: str = "documents"
    ):
        """
        Args:
            db_path: DuckDBのデータベースパス
            embedder: ベクトル化用のエンベッダー（Noneの場合は自動作成）
            table_name: ベクトルが保存されているテーブル名
        """
        self.db_path = db_path
        self.table_name = table_name

        # DuckDBの接続を作成
        self._create_db_connection()

        # ベクトルの次元数を確認
        self.vector_dim = self._get_vector_dimension()

        # 利用可能な類似度関数を確認
        self.similarity_functions = self._detect_similarity_functions()

        # エンベッダーの設定
        self.embedder = embedder if embedder is not None else get_embedder()

    def _create_db_connection(self):
        """DuckDBの接続を作成し、VSSを設定する"""
        try:
            # データベースファイルが存在することを確認
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(f"データベースファイルが見つかりません: {self.db_path}")

            # DuckDBに接続
            self.conn = duckdb.connect(self.db_path)

            # VSS拡張をインストールしてロード
            self.conn.execute("INSTALL vss")
            self.conn.execute("LOAD vss")

            logger.info(f"DuckDBへの接続とVSS拡張のロードが完了しました: {self.db_path}")

        except Exception as e:
            logger.error(f"DuckDBの設定に失敗しました: {e}")
            raise

    def _get_vector_dimension(self) -> int:
        """ベクトルの次元数を取得する"""
        try:
            # テーブルスキーマから次元数を取得
            schema = self.conn.execute(
                f"DESCRIBE {self.table_name}").fetchall()
            for col in schema:
                if col[0] == 'vector':
                    # 型名から次元数を抽出 (例: "FLOAT[2048]" -> 2048)
                    dim_str = col[1].split('[')[1].split(']')[0]
                    return int(dim_str)

            # デフォルト値
            return 2048

        except Exception as e:
            logger.warning(f"ベクトルの次元数の取得に失敗しました: {e}")
            return 2048

    def _detect_similarity_functions(self) -> Dict[str, str]:
        """利用可能な類似度関数を検出する"""
        functions = {}
        try:
            # 利用可能な関数を取得
            result = self.conn.execute(
                "SELECT function_name FROM duckdb_functions() WHERE function_name LIKE '%similarity%'"
            ).fetchall()

            function_names = [f[0] for f in result]
            logger.info(f"利用可能な類似度関数: {function_names}")

            # 利用可能な関数をマッピング
            if 'array_cosine_similarity' in function_names:
                functions['array_cosine'] = 'array_cosine_similarity'
            if 'list_cosine_similarity' in function_names:
                functions['list_cosine'] = 'list_cosine_similarity'
            if 'cosine_similarity' in function_names:
                functions['cosine'] = 'cosine_similarity'
            if 'vector_similarity' in function_names:
                functions['vector'] = 'vector_similarity'

            return functions

        except Exception as e:
            logger.error(f"類似度関数の検出に失敗しました: {e}")
            return {}

    def search(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """クエリに似たドキュメントを検索する

        Args:
            query: 検索クエリ
            top_k: 上位何件を返すか
            similarity_threshold: 類似度のしきい値

        Returns:
            検索結果のリスト
        """
        try:
            # クエリをベクトル化
            query_vector = self.embedder.embed_query(query)

            # 試行すべき検索方法
            search_methods = [
                self._search_with_dotproduct,
                self._search_with_similarity_functions,
                self._search_with_keyword,
                self._search_with_manual_calculation
            ]

            # 各方法を順番に試す
            for method in search_methods:
                try:
                    results = method(query, query_vector,
                                     top_k, similarity_threshold)
                    if results:
                        return results
                except Exception as method_error:
                    logger.warning(
                        f"検索方法 {method.__name__} が失敗しました: {method_error}")
                    continue

            # すべての方法が失敗した場合は空のリストを返す
            logger.warning("すべての検索方法が失敗しました")
            return []

        except Exception as e:
            logger.error(f"検索に失敗しました: {e}")
            return []

    def _search_with_dotproduct(self, query, query_vector, top_k, threshold):
        """内積を使用した検索"""
        logger.info("内積による検索を試みます...")

        # 内積クエリの作成（ベクトル同士の内積を計算）
        sql = f"""
            SELECT
                id,
                title,
                category,
                content,
                filename,
                (SELECT SUM(v1 * v2)
                FROM (SELECT UNNEST(vector) AS v1, UNNEST(?::FLOAT[{self.vector_dim}]) AS v2))
                / (SQRT((SELECT SUM(v * v) FROM (SELECT UNNEST(vector) AS v))) *
                SQRT((SELECT SUM(v * v) FROM (SELECT UNNEST(?::FLOAT[{self.vector_dim}]) AS v))))
                AS similarity
            FROM {self.table_name}
            ORDER BY similarity DESC
            LIMIT ?
        """

        result = self.conn.execute(
            sql, [query_vector, query_vector, top_k]).fetchall()

        # 結果を変換
        results = []
        for row in result:
            results.append({
                "id": row[0],
                "title": row[1],
                "category": row[2],
                "content": row[3],
                "filename": row[4],
                "similarity": row[5]
            })

        if results:
            logger.info(f"内積による検索で {len(results)} 件見つかりました")

        return results

    def _search_with_similarity_functions(self, query, query_vector, top_k, threshold):
        """利用可能な類似度関数による検索"""
        if not self.similarity_functions:
            return []

        # 利用可能な関数を使用
        for func_type, func_name in self.similarity_functions.items():
            try:
                logger.info(f"{func_name}による検索を試みます...")

                # 型キャストを含むSQLを準備
                sql = f"""
                    SELECT
                        id,
                        title,
                        category,
                        content,
                        filename,
                        {func_name}(vector, ?::FLOAT[{self.vector_dim}]) AS similarity
                    FROM {self.table_name}
                    ORDER BY similarity DESC
                    LIMIT ?
                """

                result = self.conn.execute(
                    sql, [query_vector, top_k]).fetchall()

                # 結果を変換
                results = []
                for row in result:
                    results.append({
                        "id": row[0],
                        "title": row[1],
                        "category": row[2],
                        "content": row[3],
                        "filename": row[4],
                        "similarity": row[5]
                    })

                if results:
                    logger.info(f"{func_name}による検索で {len(results)} 件見つかりました")
                    return results

            except Exception as e:
                logger.warning(f"{func_name}による検索に失敗しました: {e}")

        return []

    def _search_with_keyword(self, query, query_vector, top_k, threshold):
        """キーワードによる検索"""
        logger.info("キーワード検索を試みます...")

        sql = f"""
            SELECT
                id,
                title,
                category,
                content,
                filename,
                1.0 AS similarity
            FROM {self.table_name}
            WHERE content LIKE ? OR title LIKE ?
            LIMIT ?
        """

        search_term = f"%{query}%"
        result = self.conn.execute(
            sql, [search_term, search_term, top_k]).fetchall()

        # 結果を変換
        results = []
        for row in result:
            results.append({
                "id": row[0],
                "title": row[1],
                "category": row[2],
                "content": row[3],
                "filename": row[4],
                "similarity": 0.9  # キーワード検索の類似度は固定値
            })

        if results:
            logger.info(f"キーワード検索で {len(results)} 件見つかりました")

        return results

    def _search_with_manual_calculation(self, query, query_vector, top_k, threshold):
        """全件取得してPythonで計算する方法"""
        try:
            import numpy as np
            logger.info("手動ベクトル類似度計算を試みます...")

            # 全ドキュメントを取得
            all_docs = self.conn.execute(f"""
                SELECT
                    id,
                    title,
                    category,
                    content,
                    filename,
                    vector
                FROM {self.table_name}
            """).fetchall()

            # Pythonで類似度を計算
            results = []
            query_np = np.array(query_vector)

            for doc in all_docs:
                doc_vector = np.array(doc[5])

                # コサイン類似度を計算
                norm_product = np.linalg.norm(
                    doc_vector) * np.linalg.norm(query_np)
                if norm_product == 0:
                    similarity = 0
                else:
                    similarity = np.dot(doc_vector, query_np) / norm_product

                if similarity >= threshold:
                    results.append({
                        "id": doc[0],
                        "title": doc[1],
                        "category": doc[2],
                        "content": doc[3],
                        "filename": doc[4],
                        "similarity": float(similarity)
                    })

            # 類似度でソート
            results.sort(key=lambda x: x["similarity"], reverse=True)

            # 上位k件を取得
            results = results[:top_k]

            if results:
                logger.info(f"手動計算で {len(results)} 件見つかりました")

            return results

        except ImportError:
            logger.warning("NumPyがインストールされていないため、手動計算を実行できません")
            return []
        except Exception as e:
            logger.warning(f"手動計算に失敗しました: {e}")
            return []

    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """検索結果を読みやすい形式にフォーマットする

        Args:
            results: 検索結果のリスト

        Returns:
            フォーマットされた文字列
        """
        if not results:
            return "検索結果が見つかりませんでした。"

        formatted = "検索結果:\n\n"

        for i, result in enumerate(results, 1):
            # タイトルと類似度
            formatted += f"【{i}】 {result['title']
                                  } (類似度: {result['similarity']:.4f})\n"

            # カテゴリ
            formatted += f"カテゴリ: {result['category']}\n"

            # 内容（長すぎる場合は省略）
            content = result['content']
            if len(content) > 300:
                content = content[:300] + "..."
            formatted += f"{content}\n\n"

        return formatted

    def close(self):
        """データベース接続を閉じる"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logger.info("データベース接続を閉じました")


def search_documents(
    query: str,
    db_path: str = "data/vector_db.duckdb",
    top_k: int = 3,
    similarity_threshold: float = 0.5,
    print_results: bool = True
) -> List[Dict[str, Any]]:
    """ドキュメントを検索するヘルパー関数

    Args:
        query: 検索クエリ
        db_path: データベースパス
        top_k: 上位何件を返すか
        similarity_threshold: 類似度のしきい値
        print_results: 結果を出力するかどうか

    Returns:
        検索結果のリスト
    """
    try:
        # 検索器を作成
        retriever = VectorRetriever(db_path=db_path)

        # 検索を実行
        results = retriever.search(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        # 結果を出力（オプション）
        if print_results and results:
            print(retriever.format_results(results))

        # 検索器を閉じる
        retriever.close()

        return results

    except Exception as e:
        logger.error(f"ドキュメント検索に失敗しました: {e}")
        return []


if __name__ == "__main__":
    # 動作確認用コード
    db_path = "data/vector_db.duckdb"

    # サンプルクエリ
    queries = [
        "サン・ジュカールとは誰ですか？",
        "エシム王国について教えてください",
        "惑星ボーの月はいくつありますか？",
        "ブルカニウムとは何ですか？"
    ]

    # 各クエリで検索
    for query in queries:
        print(f"\n検索クエリ: {query}")
        results = search_documents(
            query=query,
            db_path=db_path,
            top_k=2,
            similarity_threshold=0.5
        )

        if not results:
            print("検索結果が見つかりませんでした。")
