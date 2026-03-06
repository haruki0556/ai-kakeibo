"""テキスト／画像から支出を抽出する（LangChain 構造化出力）. """
from __future__ import annotations

import base64
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from kakeibo.services.input_parser import UserInput

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from kakeibo.models import ExpenseCategory


# ----- 構造化出力用スキーマ -----

class ExpenseItem(BaseModel):
    """LLM が返す1件の支出（構造化出力用）. """
    amount_yen: int = Field(ge=0, description="金額（円）")
    category: str = Field(
        description="カテゴリ: food, living, fixed, misc, friend, dating, hobby, travel, other のいずれか"
    )
    memo: str = Field(default="", description="メモ（店名・品目など）")
    date: str = Field(description="支出日 YYYY-MM-DD。不明な場合は今日の日付")


class ExpenseExtraction(BaseModel):
    """抽出結果（複数件）. """
    items: List[ExpenseItem] = Field(default_factory=list, description="抽出した支出のリスト")


# ----- プロンプト -----

SYSTEM_PROMPT = """あなたは家計簿の入力アシスタントです。
ユーザーが送ったテキストまたは画像（レシート・メモなど）から、支出を抽出してください。

ルール:
- 金額は必ず円単位の整数で返す（「500円」→ 500）。
- カテゴリは次のいずれか1つ: food, living, fixed, misc, friend, dating, hobby, travel, other
  - food: 食費
  - living: 生活費（日用品など）
  - fixed: 固定費（家賃・通信など）
  - misc: 雑費
  - friend: 交際費
  - dating: デート費
  - hobby: 趣味費
  - travel: 旅行費
  - other: その他
- 日付が分からない場合は、今日の日付を YYYY-MM-DD で返す。
- 複数件ある場合はすべて items に含める。1件もない場合は items を空リストにする。
- 画像のみでテキストがない場合でも、画像内容から金額・カテゴリ・メモ・日付を読み取る。
"""


def _build_message_content(user_input: "UserInput", today_iso: str) -> List[dict]:
    """LLM に渡すメッセージの content（テキスト + 画像）を組み立てる."""
    content: List[dict] = []

    prompt_text = (
        f"今日の日付は {today_iso} です。\n"
        "以下のテキストまたは画像から支出を抽出し、金額・カテゴリ・メモ・日付を返してください。"
    )
    if user_input.has_text:
        prompt_text += f"\n\n【テキスト】\n{user_input.text.strip()}"
    else:
        prompt_text += "\n\n【画像のみ】画像の内容から支出を読み取ってください。"

    content.append({"type": "text", "text": prompt_text})

    for path in user_input.image_paths:
        raw = Path(path).read_bytes()
        b64 = base64.standard_b64encode(raw).decode("ascii")
        # MIME は jpeg をデフォルトに（png でも多くのモデルは解釈する）
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    return content


def _category_to_enum(category: str) -> ExpenseCategory:
    """LLM の文字列カテゴリを ExpenseCategory にマッピング."""
    c = (category or "").strip().lower()
    for e in ExpenseCategory:
        if e.value == c:
            return e
    return ExpenseCategory.other


def _parse_date(date_str: str, fallback: date) -> date:
    """YYYY-MM-DD を date に変換。失敗時は fallback."""
    if not date_str or not date_str.strip():
        return fallback
    try:
        return datetime.strptime(date_str.strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return fallback


async def extract_expenses(llm: Any, user_input: "UserInput") -> List[dict]:
    """
    テキスト／画像から支出を抽出する。

    Returns:
        Expense 用の辞書のリスト。各要素は amount_yen, category (ExpenseCategory), memo, spent_on (date) を持つ。
        source は呼び出し側で付与する想定。
    """
    today = date.today()
    today_iso = today.isoformat()
    content = _build_message_content(user_input, today_iso)
    messages = [HumanMessage(content=content)]

    structured_llm = llm.with_structured_output(ExpenseExtraction, method="json_schema")
    result: ExpenseExtraction = await structured_llm.ainvoke(messages)

    out: List[dict] = []
    for item in result.items or []:
        out.append({
            "amount_yen": item.amount_yen,
            "category": _category_to_enum(item.category),
            "memo": (item.memo or "").strip()[:500],
            "spent_on": _parse_date(item.date, today),
        })
    return out
