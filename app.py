import asyncio
import os
import re

import chainlit as cl
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from kakeibo.agent import create_budget_agent_graph
from kakeibo.agent.graph import run_agent_sync
from kakeibo.db import init_db, session_scope
from kakeibo.models import ExpenseCategory
from kakeibo.services.budget_service import get_budget_status, upsert_budget_setting
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

# Phase3: 浪費防止エージェント用グラフ（LangGraph）
budget_agent_graph = create_budget_agent_graph(llm)

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

# 予算確認・設定用キーワード（Phase2）
BUDGET_STATUS_KEYWORDS = ("予算", "残り", "使用可能", "あと何日", "いくら使える")
BUDGET_SET_PREFIX = ("予算設定", "予算設定:", "予算 ")

# Phase3-2: 購入相談とみなすキーワード（記録・予算以外を振り分け）
PURCHASE_CONSULT_KEYWORDS = ("買いたい", "買っていい", "買ってもいい", "購入していい", "買っていい？", "いい？", "買う？", "買おうか")


def _parse_amount_yen(s: str) -> int:
    """「30万」「300000」などを円で返す."""
    s = (s or "").strip().replace(",", "").replace(" ", "")
    m = re.match(r"^(\d+)(万)?$", s)
    if not m:
        return 0
    num = int(m.group(1))
    if m.group(2) == "万":
        num *= 10_000
    return num


def _parse_budget_command(text: str) -> tuple[int, int] | None:
    """
    「予算設定 25 300000」「予算 25 30万」などをパース。
    Returns (payday_day, target_amount_yen) or None.
    """
    t = (text or "").strip()
    for prefix in BUDGET_SET_PREFIX:
        if t.lower().startswith(prefix.lower()):
            rest = t[len(prefix):].strip()
            parts = rest.split()
            if len(parts) >= 2:
                try:
                    day = int(parts[0])
                    if 1 <= day <= 31:
                        yen = _parse_amount_yen(parts[1]) if len(parts[1]) > 0 else 0
                        if yen >= 0:
                            return (day, yen)
                except ValueError:
                    pass
            break
    return None


def _format_budget_status(status: dict) -> str:
    """2-4: 残り日数・使用可能金額の表示ブロック用テキスト."""
    lines = ["**予算の状況**", ""]
    lines.append(f"・今月の支出: {status['spent_yen']:,}円")
    lines.append(f"・次の給与日まで: あと **{status['remaining_days']}日**")
    if status["is_set"] and status["available_yen"] is not None:
        lines.append(f"・目標（給与日まで）: {status['target_amount_yen']:,}円")
        lines.append(f"・**使用可能金額: {status['available_yen']:,}円**")
    else:
        lines.append("・予算を設定すると「使用可能金額」が表示されます。")
        lines.append("　例: `予算設定 25 300000`（給与日25日、目標30万円）")
    return "\n".join(lines)


def _is_budget_status_request(text: str) -> bool:
    """予算状況の表示要求かどうか."""
    t = (text or "").strip()
    return any(kw in t for kw in BUDGET_STATUS_KEYWORDS) and "予算設定" not in t


def _is_purchase_consult(text: str) -> bool:
    """購入相談（〇〇買っていい？）とみなすかどうか。Phase3-2 入力ルーティング."""
    t = (text or "").strip()
    if not t or len(t) > 200:
        return False
    return any(kw in t for kw in PURCHASE_CONSULT_KEYWORDS)


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

    # 直近の記録と予算状況を取得（2-4: ウェルカム時に表示）
    def _fetch_welcome_data() -> tuple:
        with session_scope() as session:
            recent = get_recent_expenses(session, user_id, limit=5)
            status = get_budget_status(session, user_id)
            return (recent, status)

    recent, budget_status = await asyncio.to_thread(_fetch_welcome_data)

    welcome = "家計簿へようこそ。支出のテキストやレシート画像を送ると記録します。\n\n"
    welcome += _format_budget_status(budget_status)
    welcome += "\n\n"
    welcome += _format_expense_list(recent, title="最近の記録（直近5件）")

    # 2-3: 予算を見る / 予算を設定 アクション
    actions = [
        cl.Action(name="show_budget", label="予算を見る", payload={}),
        cl.Action(name="setup_budget", label="予算を設定", payload={}),
    ]
    await cl.Message(content=welcome, actions=actions).send()


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


def _fetch_budget_status_sync(user_id: int) -> dict:
    """同期で予算状況を取得（スレッド用）. """
    with session_scope() as session:
        return get_budget_status(session, user_id)


@cl.action_callback("show_budget")
async def on_show_budget(action: cl.Action) -> None:
    """2-3: 「予算を見る」ボタン → 残り日数・使用可能金額を表示."""
    user_id = await _resolve_user_id()
    status = await asyncio.to_thread(_fetch_budget_status_sync, user_id)
    await cl.Message(content=_format_budget_status(status)).send()
    await action.remove()


@cl.action_callback("setup_budget")
async def on_setup_budget(action: cl.Action) -> None:
    """2-3: 「予算を設定」ボタン → 入力例を案内."""
    await cl.Message(
        content="予算を設定するには、次のように送信してください。\n\n"
        "**予算設定 給与日 目標額（円）**\n\n"
        "例: `予算設定 25 300000`（毎月25日が給与日、給与日までに30万円以内に抑えたい場合）\n"
        "例: `予算設定 25 30万`"
    ).send()
    await action.remove()


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

    # 2-3: 予算設定コマンド（例: 予算設定 25 300000）
    if user_input.has_text and not user_input.has_images:
        parsed = _parse_budget_command(user_input.text)
        if parsed is not None:
            payday_day, target_yen = parsed
            user_id = await _resolve_user_id()

            def _upsert() -> dict:
                with session_scope() as session:
                    upsert_budget_setting(session, user_id, payday_day, target_yen)
                    return get_budget_status(session, user_id)

            status = await asyncio.to_thread(_upsert)
            await cl.Message(
                content=f"予算を設定しました（給与日{payday_day}日、目標{target_yen:,}円）。\n\n" + _format_budget_status(status)
            ).send()
            return

    # 2-4: 予算状況の表示要求（「予算」「残り」「使用可能」など）
    if user_input.has_text and not user_input.has_images and _is_budget_status_request(user_input.text):
        user_id = await _resolve_user_id()
        status = await asyncio.to_thread(_fetch_budget_status_sync, user_id)
        await cl.Message(content=_format_budget_status(status)).send()
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

