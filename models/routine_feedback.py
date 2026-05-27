from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Column, DateTime, Integer, JSON, String, Text
from sqlalchemy.sql import func

from database import Base


class RoutineFeedback(Base):
    __tablename__ = "routine_feedback"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(128), nullable=True, index=True)
    routine_draft_id = Column(String(128), nullable=False, index=True)
    rating = Column(Integer, nullable=True)
    completed = Column(Boolean, nullable=True)
    accepted_without_edits = Column(Boolean, nullable=True)
    skipped_exercise_ids = Column(JSON, nullable=True)
    edited_exercises = Column(JSON, nullable=True)
    user_note = Column(Text(500), nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    source = Column(String(32), nullable=False, default="api")
