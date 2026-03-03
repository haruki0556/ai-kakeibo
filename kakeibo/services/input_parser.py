"""Chainlit メッセージからテキストと画像を抽出する（Phase1 入力受付）. """
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List


@dataclass
class UserInput:
    """ユーザーが送った入力（テキスト＋画像）のまとめ."""

    text: str
    """テキスト本文（空文字の可能性あり）. """
    image_paths: List[str]
    """添付画像のローカルパス一覧（レシート写真など）. """

    @property
    def has_images(self) -> bool:
        return len(self.image_paths) > 0

    @property
    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())


def parse_message(message: Any) -> UserInput:
    """
    Chainlit のメッセージからテキストと画像パスを抽出する。

    - テキスト: message.content
    - 画像: message.elements のうち type が image の要素の path を収集
    """
    text = (message.content or "").strip()
    image_paths: List[str] = []

    elements = getattr(message, "elements", None) or []
    for el in elements:
        # type が image のもの（レシート等）を対象
        el_type = getattr(el, "type", None)
        if el_type == "image":
            path = getattr(el, "path", None)
            if path and Path(path).exists():
                image_paths.append(str(path))

    return UserInput(text=text, image_paths=image_paths)
