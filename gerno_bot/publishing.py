"""Make sure every forecast the bot posts also gets its comment.

forecasting-tools posts the forecast first and the comment second. If the
comment call fails after the forecast went in, the next run would skip the
question as already forecast and the comment would never be posted, which
would break the prize rule that bots comment on every question they forecast.
This wrapper saves any comment that failed to post and retries it at the
start of the next run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CommentSafeClient:
    def __init__(self, inner: Any, pending_path: str | Path) -> None:
        self._inner = inner
        self._pending_path = Path(pending_path)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    # ---------------------------------------------------------------- storage
    def _load(self) -> list[dict]:
        if not self._pending_path.exists():
            return []
        with self._pending_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, items: list[dict]) -> None:
        self._pending_path.parent.mkdir(parents=True, exist_ok=True)
        with self._pending_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(items, f, indent=2)
            f.write("\n")

    def pending_count(self) -> int:
        return len(self._load())

    # ---------------------------------------------------------------- posting
    def post_question_comment(self, post_id: int, comment_text: str, **kwargs: Any) -> None:
        try:
            self._inner.post_question_comment(post_id, comment_text, **kwargs)
        except Exception:
            items = self._load()
            items.append({"post_id": post_id, "text": comment_text})
            self._save(items)
            logger.error("Comment on post %s failed; saved for retry next run", post_id)
            raise

    def retry_pending(self) -> int:
        """Post saved comments. Returns how many are still pending."""
        items = self._load()
        if not items:
            return 0
        still_pending: list[dict] = []
        for item in items:
            try:
                self._inner.post_question_comment(item["post_id"], item["text"])
                logger.info("Posted saved comment on post %s", item["post_id"])
            except Exception as e:
                logger.error("Saved comment on post %s failed again: %s", item["post_id"], e)
                still_pending.append(item)
        self._save(still_pending)
        return len(still_pending)
