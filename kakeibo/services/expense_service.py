"""支出の保存とデフォルトユーザー取得（Phase1 抽出・分類）. """
from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING, List

from sqlmodel import Session, select

from kakeibo.models import Expense, User
from kakeibo.models import ExpenseCategory


def sum_expenses_by_category_for_month(
    session: Session,
    user_id: int,
    year_month: str,
) -> dict:
    """指定月（YYYY-MM）のカテゴリ別支出合計。ExpenseCategory -> 円."""
    parts = year_month.split("-")
    if len(parts) != 2:
        return {}
    y, m = int(parts[0]), int(parts[1])
    start = date(y, m, 1)
    if m == 12:
        end = date(y, 12, 31)
    else:
        end = date(y, m + 1, 1) - timedelta(days=1)
    stmt = (
        select(Expense.category, Expense.amount_yen)
        .where(Expense.user_id == user_id)
        .where(Expense.spent_on >= start)
        .where(Expense.spent_on <= end)
    )
    rows = session.exec(stmt).all()
    result: dict = {}
    for cat, amt in rows:
        result[cat] = result.get(cat, 0) + amt
    return result


def get_recent_expenses(session: Session, user_id: int, limit: int = 10) -> List[Expense]:
    """指定ユーザーの直近の支出を新しい順で取得する."""
    stmt = (
        select(Expense)
        .where(Expense.user_id == user_id)
        .order_by(Expense.created_at.desc())
        .limit(limit)
    )
    return list(session.exec(stmt).all())


if TYPE_CHECKING:
    from kakeibo.models import ExpenseCategory


def get_or_create_default_user(session: Session) -> User:
    """id=1 のユーザーを取得。いなければ name=default で作成する."""
    user = session.get(User, 1)
    if user is not None:
        return user
    user = User(id=1, name="default")
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def save_expenses(
    session: Session,
    user_id: int,
    items: List[dict],
    source: str = "text",
) -> List[Expense]:
    """
    抽出結果を expenses に保存する。

    items: 各要素は amount_yen, category (ExpenseCategory), memo, spent_on (date) を持つ辞書。
    """
    created: List[Expense] = []
    for d in items:
        expense = Expense(
            user_id=user_id,
            amount_yen=d["amount_yen"],
            category=d["category"],
            memo=d.get("memo") or "",
            spent_on=d["spent_on"],
            source=source,
        )
        session.add(expense)
        created.append(expense)
    session.commit()
    for e in created:
        session.refresh(e)
    return created
