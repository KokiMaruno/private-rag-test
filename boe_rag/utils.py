#!/usr/bin/env python
"""
ユーティリティ関数を提供するモジュール
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_dir(directory: str) -> bool:
    """ディレクトリが存在することを確認し、存在しなければ作成する

    Args:
        directory: 確認するディレクトリパス

    Returns:
        成功したかどうか
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"ディレクトリの作成に失敗しました: {e}")
        return False


def get_text_files(directory: str) -> List[str]:
    """指定したディレクトリ内のテキストファイルのパスのリストを返す

    Args:
        directory: 探索するディレクトリパス

    Returns:
        テキストファイルのパスのリスト
    """
    try:
        if not os.path.exists(directory):
            logger.warning(f"ディレクトリが存在しません: {directory}")
            return []

        text_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".txt"):
                    text_files.append(os.path.join(root, file))

        return text_files

    except Exception as e:
        logger.error(f"テキストファイルの取得に失敗しました: {e}")
        return []


def read_text_file(file_path: str) -> str:
    """テキストファイルの内容を読み込む

    Args:
        file_path: ファイルパス

    Returns:
        ファイルの内容
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    except Exception as e:
        logger.error(f"ファイルの読み込みに失敗しました: {e}")
        return ""


def evaluate_search_results(
    results: List[Dict[str, Any]],
    expected_texts: List[str],
    case_sensitive: bool = False
) -> Dict[str, float]:
    """検索結果を評価する

    Args:
        results: 検索結果のリスト
        expected_texts: 期待されるテキストのリスト
        case_sensitive: 大文字小文字を区別するかどうか

    Returns:
        評価結果の辞書（precision, recallなど）
    """
    if not results or not expected_texts:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "hit_rate": 0.0
        }

    # 検索結果のテキスト
    result_texts = [result["content"] for result in results]

    # 大文字小文字を区別しない場合
    if not case_sensitive:
        result_texts = [text.lower() for text in result_texts]
        expected_texts = [text.lower() for text in expected_texts]

    # ヒットしたアイテムの数を計算
    hits = 0
    for expected in expected_texts:
        for result in result_texts:
            if expected in result:
                hits += 1
                break

    # 評価指標を計算
    precision = hits / len(result_texts) if result_texts else 0
    recall = hits / len(expected_texts) if expected_texts else 0
    f1_score = 2 * precision * recall / \
        (precision + recall) if precision + recall > 0 else 0
    hit_rate = hits / len(expected_texts) if expected_texts else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "hit_rate": hit_rate
    }


def format_evaluation(evaluation: Dict[str, float]) -> str:
    """評価結果を読みやすい形式にフォーマットする

    Args:
        evaluation: evaluate_search_resultsの戻り値

    Returns:
        フォーマットされた文字列
    """
    return (
        f"Precision: {evaluation['precision']:.4f}\n"
        f"Recall: {evaluation['recall']:.4f}\n"
        f"F1 Score: {evaluation['f1_score']:.4f}\n"
        f"Hit Rate: {evaluation['hit_rate']:.4f}"
    )
