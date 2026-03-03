import os
import chainlit as cl
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from kakeibo.db import init_db
from kakeibo.services.input_parser import parse_message, UserInput


# .env から環境変数を読み込む
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("環境変数 GOOGLE_API_KEY が見つかりません。.env を確認してください。")


# Gemini 2.5 Flash を使う LLM クライアントを作成
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=GOOGLE_API_KEY,
)


@cl.on_chat_start
async def on_chat_start() -> None:
    # 初回起動時にDBテーブルを作成
    init_db()


def _format_receipt_confirmation(user_input: UserInput) -> str:
    """入力受付の確認文を作る."""
    parts = []
    if user_input.has_text:
        parts.append(f"テキスト（{len(user_input.text)}文字）")
    if user_input.has_images:
        parts.append(f"画像{len(user_input.image_paths)}枚")
    if not parts:
        return "テキストまたは画像を送信してください。"
    return "".join(parts) + "を受け取りました。次のステップで分類・記録します。"


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """テキストまたはレシート画像の入力を受け付ける（Phase1 入力受付）. """
    user_input = parse_message(message)
    confirmation = _format_receipt_confirmation(user_input)
    await cl.Message(content=confirmation).send()

