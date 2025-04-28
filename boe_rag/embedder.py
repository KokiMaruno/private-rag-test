#!/usr/bin/env python
"""
テキストをベクトル化するモジュール
Plamo-Embedding-1Bを使用
"""
import os
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Union, Dict, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlamoEmbedder:
    """Plamo-Embedding-1Bを使ったテキストエンベッダー"""

    def __init__(self, cache_dir: str = None):
        """
        Args:
            cache_dir: モデルのキャッシュディレクトリ
        """
        logger.info("Plamo-Embedding-1Bの初期化を開始します…")

        # モデル名
        self.model_name = "pfnet/plamo-embedding-1b"

        # デバイスの設定(M1 Macの場合はMPSを利用)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("MPS (Metal Performance Shaders)を使用します")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("CUDAを使用します")
        else:
            self.device = torch.device("cpu")
            logger.info("CPUを使用します")

        # トークナイザーとモデルの読み込み
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            # モデルをデバイスに移動
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Plamo-Embedding-1Bの初期化が完了しました。デバイス: {self.device}")

        except Exception as e:
            logger.error(f"モデルの読み込みに失敗しました: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """テキストをベクトル化する

        Args:
            text: ベクトル化するテキスト

        Returns:
            ベクトル（浮動小数点数のリスト）
        """
        try:
            # 推論モードで実行
            with torch.inference_mode():
                # モデルのencode_documentメソッドを使用
                # これはPlamo-Embedding-1Bの特殊なメソッド
                embeddings = self.model.encode_document([text], self.tokenizer)

                # 最初のテキストのベクトルを取得し、CPU上のPythonのリストに変換
                vector = embeddings[0].cpu().tolist()

                return vector

        except Exception as e:
            logger.error(f"ベクトル化に失敗しました: {e}")
            # エラーの場合は空のベクトルを返す
            return []

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """複数のドキュメントをベクトル化する

        Args:
            documents: ベクトル化するドキュメントのリスト

        Returns:
            ベクトルのリスト
        """
        try:
            # 推論モードで実行
            with torch.inference_mode():
                # バッチサイズを設定（メモリ効率のため）
                batch_size = 16
                all_embeddings = []

                # バッチ処理でドキュメントをベクトル化
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i+batch_size]
                    logger.info(
                        f"バッチ {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} を処理中...")

                    # モデルのencode_documentメソッドを使用
                    batch_embeddings = self.model.encode_document(
                        batch, self.tokenizer)

                    # CPU上のPythonのリストに変換
                    batch_vectors = [emb.cpu().tolist()
                                     for emb in batch_embeddings]
                    all_embeddings.extend(batch_vectors)

                return all_embeddings

        except Exception as e:
            logger.error(f"バッチベクトル化に失敗しました: {e}")
            # エラーの場合は空のリストを返す
            return []

    def embed_query(self, query: str) -> List[float]:
        """検索クエリをベクトル化する
        ドキュメントとクエリで異なる処理をする場合があるため分けています

        Args:
            query: 検索クエリ

        Returns:
            クエリベクトル
        """
        # クエリのベクトル化（現在はドキュメントと同じ処理）
        return self.embed_text(query)


def get_embedder(cache_dir: str = None) -> PlamoEmbedder:
    """エンベッダーのインスタンスを取得する

    Args:
        cache_dir: モデルのキャッシュディレクトリ

    Returns:
        PlamoEmbedder インスタンス
    """
    return PlamoEmbedder(cache_dir=cache_dir)


if __name__ == "__main__":
    # 動作確認用コード
    embedder = get_embedder()

    # サンプルテキスト
    sample_text = "サン・ジュカールは惑星ボーの英雄です。"
    sample_query = "誰が惑星ボーの英雄ですか？"

    # ベクトル化
    text_vector = embedder.embed_text(sample_text)
    query_vector = embedder.embed_query(sample_query)

    # ベクトルのサイズを確認
    print(f"テキストベクトルのサイズ: {len(text_vector)}")
    print(f"クエリベクトルのサイズ: {len(query_vector)}")

    # ベクトルの一部を表示
    print(f"テキストベクトルの先頭10要素: {text_vector[:10]}")
    print(f"クエリベクトルの先頭10要素: {query_vector[:10]}")
