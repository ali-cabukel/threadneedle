from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: str = Field(default="default")
    regenerate: bool = False


class ChatResponse(BaseModel):
    thread_id: str
    response: str
    citations: list[dict] = Field(default_factory=list)
