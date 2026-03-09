"""予算設定の保存と残り日数・使用可能金額の計算（Phase2）. """
from __future__ import annotations

from calendar import monthrange
from datetime import date, datetime
from typing import Optional

from sqlmodel import Session, select

from kakeibo.models import BudgetSetting, Expense


# ----- 2-1: 設定の保存 -----


def get_budget_setting(session: Session, user_id: int) -> Optional[BudgetSetting]:
    """ユーザーの予算設定を1件取得する。無ければ None."""
    stmt = select(BudgetSetting).where(BudgetSetting.user_id == user_id).limit(1)
    return session.exec(stmt).first()


def upsert_budget_setting(
    session: Session,
    user_id: int,
    payday_day: int,
    target_amount_yen: int,
) -> BudgetSetting:
    """
    予算設定を1ユーザー1行で更新する。無ければ作成。
    payday_day: 毎月の給与日（1-31）
    target_amount_yen: 給与日までに抑えたい支出（円）
    """
    row = get_budget_setting(session, user_id)
    if row is None:
        row = BudgetSetting(
            user_id=user_id,
            payday_day=payday_day,
            target_amount_yen=target_amount_yen,
        )
        session.add(row)
    else:
        row.payday_day = payday_day
        row.target_amount_yen = target_amount_yen
        row.updated_at = datetime.utcnow()
    session.commit()
    session.refresh(row)
    return row


# ----- 2-2: 計算ロジック -----


def _next_payday(today: date, payday_day: int) -> date:
    """今日から見た「次の給与日」の日付。今日が給与日なら今日を返す."""
    year, month = today.year, today.month
    last = monthrange(year, month)[1]
    day_this_month = min(payday_day, last)
    pay = date(year, month, day_this_month)
    if today <= pay:
        return pay
    # 来月の給与日
    if month == 12:
        year, month = year + 1, 1
    else:
        month += 1
    last = monthrange(year, month)[1]
    return date(year, month, min(payday_day, last))


def remaining_days_until_payday(today: date, payday_day: int) -> int:
    """今日から次の給与日までの残り日数（給与日当日は0）. """
    next_p = _next_payday(today, payday_day)
    return max(0, (next_p - today).days)


def sum_expenses_between(
    session: Session,
    user_id: int,
    start_date: date,
    end_date: date,
) -> int:
    """指定期間内の支出合計（円）を返す。spent_on で集計."""
    stmt = (
        select(Expense.amount_yen)
        .where(Expense.user_id == user_id)
        .where(Expense.spent_on >= start_date)
        .where(Expense.spent_on <= end_date)
    )
    rows = session.exec(stmt).all()
    return sum(rows) if rows else 0


def spent_from_month_start(session: Session, user_id: int, today: date) -> int:
    """今月1日〜今日までの支出合計（円）. """
    start = date(today.year, today.month, 1)
    return sum_expenses_between(session, user_id, start, today)


def get_budget_status(
    session: Session,
    user_id: int,
    today: Optional[date] = None,
) -> dict:
    """
    残り日数・今月支出・目標額・使用可能金額をまとめて返す。

    Returns:
        is_set: 予算設定が存在し目標額が設定されているか
        payday_day: 給与日（日）
        target_amount_yen: 目標支出額（円）
        spent_yen: 今月1日〜今日の支出合計（円）
        remaining_days: 次の給与日までの残り日数
        available_yen: 使用可能金額（目標 - 今月支出）。未設定なら None
        next_payday: 次の給与日の日付（表示用）
    """
    if today is None:
        today = date.today()

    setting = get_budget_setting(session, user_id)
    if setting is None:
        return {
            "is_set": False,
            "payday_day": 25,
            "target_amount_yen": 0,
            "spent_yen": spent_from_month_start(session, user_id, today),
            "remaining_days": remaining_days_until_payday(today, 25),
            "available_yen": None,
            "next_payday": _next_payday(today, 25),
        }

    spent = spent_from_month_start(session, user_id, today)
    remaining = remaining_days_until_payday(today, setting.payday_day)
    target = setting.target_amount_yen
    available = (target - spent) if target > 0 else None

    return {
        "is_set": target > 0,
        "payday_day": setting.payday_day,
        "target_amount_yen": target,
        "spent_yen": spent,
        "remaining_days": remaining,
        "available_yen": available,
        "next_payday": _next_payday(today, setting.payday_day),
    }
