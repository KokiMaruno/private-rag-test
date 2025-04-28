#!/usr/bin/env python
"""
RAGシステムのメインスクリプト
ドキュメントのインデックス作成と検索を行う
"""
from boe_rag.utils import ensure_dir
from boe_rag.retriever import search_documents
from boe_rag.indexer import create_index_from_directory
import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# モジュールのインポート


def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(description="惑星ボー神話RAGシステム")

    # サブコマンドの設定
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンド")

    # インデックス作成コマンド
    index_parser = subparsers.add_parser("index", help="ドキュメントのインデックスを作成")
    index_parser.add_argument(
        "--data-dir", default="data/boe_mythology", help="データディレクトリ")
    index_parser.add_argument(
        "--db-path", default="data/vector_db.duckdb", help="データベースパス")
    index_parser.add_argument(
        "--force", action="store_true", help="既存のインデックスを上書き")

    # 検索コマンド
    search_parser = subparsers.add_parser("search", help="ドキュメントを検索")
    search_parser.add_argument("--query", required=True, help="検索クエリ")
    search_parser.add_argument(
        "--db-path", default="data/vector_db.duckdb", help="データベースパス")
    search_parser.add_argument("--top-k", type=int, default=3, help="上位何件を返すか")
    search_parser.add_argument(
        "--threshold", type=float, default=0.5, help="類似度のしきい値")

    # 対話モードコマンド
    interactive_parser = subparsers.add_parser("interactive", help="対話モードで検索")
    interactive_parser.add_argument(
        "--db-path", default="data/vector_db.duckdb", help="データベースパス")
    interactive_parser.add_argument(
        "--top-k", type=int, default=3, help="上位何件を返すか")
    interactive_parser.add_argument(
        "--threshold", type=float, default=0.5, help="類似度のしきい値")

    # データ生成コマンド
    gen_data_parser = subparsers.add_parser("generate-data", help="サンプルデータを生成")
    gen_data_parser.add_argument(
        "--output-dir", default="data/boe_mythology", help="出力ディレクトリ")

    return parser.parse_args()


def generate_sample_data(output_dir: str):
    """サンプルデータを生成する

    Args:
        output_dir: 出力ディレクトリ
    """
    # 出力ディレクトリの作成
    if not ensure_dir(output_dir):
        logger.error(f"出力ディレクトリの作成に失敗しました: {output_dir}")
        return False

    try:
        # サンプルデータ生成スクリプトの実行
        # boe_mythology_dataというパッケージが存在するか確認
        try:
            from boe_mythology_data import save_mythology_data
            save_mythology_data(output_dir)
            return True
        except ImportError:
            # スクリプトを直接実行
            script_path = Path(__file__).parent / "scripts" / \
                "generate_boe_mythology.py"
            if script_path.exists():
                import subprocess
                result = subprocess.run(
                    ["python", str(script_path), "--output-dir", output_dir])
                return result.returncode == 0
            else:
                # スクリプトが見つからない場合は内部モジュールを使用
                from scripts.generate_boe_mythology import save_mythology_data
                save_mythology_data(output_dir)
                return True

    except Exception as e:
        logger.error(f"サンプルデータの生成に失敗しました: {e}")
        return False


def interactive_mode(db_path: str, top_k: int, similarity_threshold: float):
    """対話モードで検索を行う

    Args:
        db_path: データベースパス
        top_k: 上位何件を返すか
        similarity_threshold: 類似度のしきい値
    """
    print("\n===== 惑星ボー神話RAG検索システム（対話モード） =====")
    print("質問を入力してください（終了するには 'q' または 'quit' を入力）\n")

    while True:
        # ユーザー入力を取得
        query = input("\n質問: ")

        # 終了条件のチェック
        if query.lower() in ("q", "quit", "exit"):
            print("システムを終了します。")
            break

        # 空の入力をスキップ
        if not query.strip():
            continue

        # 検索を実行
        results = search_documents(
            query=query,
            db_path=db_path,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            print_results=True
        )

        # 検索結果がない場合
        if not results:
            print("検索結果が見つかりませんでした。別の質問を試してください。")


def main():
    """メイン関数"""
    # コマンドライン引数のパース
    args = parse_args()

    # コマンドの処理
    if args.command == "index":
        # インデックス作成
        logger.info(f"ディレクトリ {args.data_dir} からインデックスを作成します...")
        success = create_index_from_directory(
            data_dir=args.data_dir,
            db_path=args.db_path,
            drop_existing=args.force
        )

        if success:
            logger.info("インデックス作成が完了しました！")
        else:
            logger.error("インデックス作成に失敗しました。")

    elif args.command == "search":
        # 検索
        logger.info(f"クエリ '{args.query}' で検索します...")
        results = search_documents(
            query=args.query,
            db_path=args.db_path,
            top_k=args.top_k,
            similarity_threshold=args.threshold
        )

        if not results:
            logger.warning("検索結果が見つかりませんでした。")

    elif args.command == "interactive":
        # 対話モード
        interactive_mode(
            db_path=args.db_path,
            top_k=args.top_k,
            similarity_threshold=args.threshold
        )

    elif args.command == "generate-data":
        # サンプルデータの生成
        logger.info(f"サンプルデータを {args.output_dir} に生成します...")
        success = generate_sample_data(args.output_dir)

        if success:
            logger.info("サンプルデータの生成が完了しました！")
        else:
            logger.error("サンプルデータの生成に失敗しました。")

    else:
        # コマンドが指定されていない場合
        logger.error(
            "コマンドを指定してください（index, search, interactive, generate-data）")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
