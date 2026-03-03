from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from sqlmodel import Session, SQLModel, create_engine


def _database_url() -> str:
    # 例: sqlite:///./kakeibo.db
    return os.getenv("DATABASE_URL", "sqlite:///./kakeibo.db")


engine = create_engine(
    _database_url(),
    echo=os.getenv("SQL_ECHO", "0") == "1",
    connect_args={"check_same_thread": False},
)


def init_db() -> None:
    """テーブルを作成（SQLite 前提の簡易初期化）。"""
    # モデルを import して metadata に登録させる
    from . import models as _models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    #メモ：modelsに登録していたすべてのテーブルを作成する


@contextmanager
def session_scope() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
        #yeildで一時的にセッションを開き、ブロックが終わると自動的にセッションを閉じる
        #これにより、with文のブロックが終わると自動的にセッションが閉じられる

