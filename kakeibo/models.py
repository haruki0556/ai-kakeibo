from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


class ExpenseCategory(str, Enum):
    food = "food"  # 食費
    living = "living"  # 生活費（日用品など）
    fixed = "fixed"  # 固定費（家賃・通信など）
    misc = "misc"  # 雑費
    friend = "friend"  # 交際費
    dating = "dating"  # デート費
    hobby = "hobby"  # 趣味費
    travel = "travel"  # 旅行費
    other = "other"  # その他

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(default="default")
    line_user_id: Optional[str] = Field(default=None, index=True, unique=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class BudgetSetting(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)

    # 毎月の給与日（1-31）
    payday_day: int = Field(default=25, ge=1, le=31)

    # 給与日までに抑えたい支出（円）
    target_amount_yen: int = Field(default=0, ge=0)

    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class Expense(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)

    amount_yen: int = Field(ge=0, index=True)
    category: ExpenseCategory = Field(index=True)

    # ユーザーが入力した補足（例: 店名・品目）
    memo: str = Field(default="")

    # 支出日（不明な場合は当日として扱う運用を想定）
    spent_on: date = Field(index=True)

    # 入力ソース（text/receipt_image など）
    source: str = Field(default="text", index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

