from pydantic import BaseModel, Field


class ApprovalRequest(BaseModel):
    reviewer: str = Field(min_length=2, max_length=120)
    note: str = Field(default="", max_length=500)


class AssistantRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class AssistantResponse(BaseModel):
    answer: str
    sources: list[dict]
    mode: str
