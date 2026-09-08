"""Shared data contracts."""

from pydantic import BaseModel


class DocChunk(BaseModel):
    id: str
    doc_id: str
    heading: str
    text: str


class SearchResult(BaseModel):
    chunk: DocChunk
    score: float


class AnswerJudgeResult(BaseModel):
    correct: bool
    explanation: str
