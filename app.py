import os

import chainlit as cl
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from kakeibo.db import init_db


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


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """ユーザーからのテキストメッセージに Gemini が返信するだけのシンプルなハンドラ。"""
    # LangChain の ChatGoogleGenerativeAI は非同期メソッド ainvoke を提供している
    response = await llm.ainvoke(message.content)

    # response は AIMessage なので .content からテキストを取り出す
    await cl.Message(content=str(response.content)).send()

