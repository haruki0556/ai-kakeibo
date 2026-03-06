import asyncio
import os

import chainlit as cl
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from kakeibo.db import init_db, session_scope
from kakeibo.models import ExpenseCategory
from kakeibo.services.expense_extractor import extract_expenses
from kakeibo.services.expense_service import (
    get_or_create_default_user,
    get_recent_expenses,
    save_expenses,
)
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

# カテゴリの表示名（返答用）
CATEGORY_LABELS = {
    ExpenseCategory.food: "食費",
    ExpenseCategory.living: "生活費",
    ExpenseCategory.fixed: "固定費",
    ExpenseCategory.misc: "雑費",
    ExpenseCategory.friend: "交際費",
    ExpenseCategory.dating: "デート費",
    ExpenseCategory.hobby: "趣味費",
    ExpenseCategory.travel: "旅行費",
    ExpenseCategory.other: "その他",
}


# 「記録一覧を見る」と解釈するキーワード（部分一致）
LIST_REQUEST_KEYWORDS = ("一覧", "記録", "今日の支出", "最近の記録", "見せて", "確認")


def _is_list_request(text: str) -> bool:
    """記録一覧の表示要求かどうか."""
    t = (text or "").strip()
    return any(kw in t for kw in LIST_REQUEST_KEYWORDS) or t in ("一覧", "記録")


def _format_expense_list(expenses: list, title: str = "最近の記録") -> str:
    """支出リストを表示用テキストにする."""
    if not expenses:
        return f"**{title}**\nまだ記録がありません。"
    lines = [f"**{title}**", ""]
    for e in expenses:
        label = CATEGORY_LABELS.get(e.category, e.category.value)
        lines.append(f"・{e.spent_on.isoformat()} {e.amount_yen}円（{label}）{f' - {e.memo}' if e.memo else ''}")
    return "\n".join(lines)


@cl.on_chat_start
async def on_chat_start() -> None:
    # 初回起動時にDBテーブルを作成
    init_db()

    # ユーザー識別: セッションに user_id を保存（単一ユーザー前提で id=1）
    user_id = await asyncio.to_thread(_ensure_user_id)
    cl.user_session.set("user_id", user_id)

    # Chainlit 上で記録を確認できるよう、直近の記録をウェルカムで表示
    def _fetch_recent() -> list:
        with session_scope() as session:
            return get_recent_expenses(session, user_id, limit=5)

    recent = await asyncio.to_thread(_fetch_recent)
    welcome = "家計簿へようこそ。支出のテキストやレシート画像を送ると記録します。\n\n"
    welcome += _format_expense_list(recent, title="最近の記録（直近5件）")
    await cl.Message(content=welcome).send()


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


def _format_recorded_reply(expenses: list) -> str:
    """保存した支出を報告する文を作る."""
    if not expenses:
        return "支出を読み取れませんでした。金額・日付・内容を書いてもう一度送ってください。"
    lines = []
    for e in expenses:
        label = CATEGORY_LABELS.get(e.category, e.category.value)
        lines.append(f"・{e.amount_yen}円（{label}）{f' - {e.memo}' if e.memo else ''}")
    return "記録しました。\n" + "\n".join(lines)


def _ensure_user_id() -> int:
    """DB でデフォルトユーザーを確保し、その id を返す（スレッド内で呼ぶ用）. """
    with session_scope() as session:
        user = get_or_create_default_user(session)
        return user.id


async def _resolve_user_id() -> int:
    """セッションから user_id を取得。無ければ DB で確保してセッションにセット."""
    user_id = cl.user_session.get("user_id")
    if user_id is not None:
        return user_id
    user_id = await asyncio.to_thread(_ensure_user_id)
    cl.user_session.set("user_id", user_id)
    return user_id


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """テキストまたはレシート画像から支出を抽出して保存、または記録一覧を表示（Phase1 Chainlit フロー）. """
    user_input = parse_message(message)

    if not user_input.has_text and not user_input.has_images:
        await cl.Message(content=_format_receipt_confirmation(user_input)).send()
        return

    # 記録一覧の表示要求（画像なしでキーワード一致）→ 一覧を返す
    if user_input.has_text and not user_input.has_images and _is_list_request(user_input.text):
        user_id = await _resolve_user_id()
        def _fetch() -> list:
            with session_scope() as session:
                return get_recent_expenses(session, user_id, limit=10)

        recent = await asyncio.to_thread(_fetch)
        await cl.Message(content=_format_expense_list(recent, title="記録一覧（直近10件）")).send()
        return

    # 抽出（LLM 構造化出力）
    try:
        items = await extract_expenses(llm, user_input)
    except Exception as e:
        await cl.Message(
            content=f"抽出中にエラーが発生しました: {e!s}。内容を書き直して再送してください。"
        ).send()
        return

    if not items:
        await cl.Message(content=_format_recorded_reply([])).send()
        return

    # DB 保存（user_id はセッションから取得）
    user_id = await _resolve_user_id()
    source = "receipt_image" if user_input.has_images else "text"

    def _save() -> list:
        with session_scope() as session:
            return save_expenses(session, user_id, items, source=source)

    saved = await asyncio.to_thread(_save)
    await cl.Message(content=_format_recorded_reply(saved)).send()

