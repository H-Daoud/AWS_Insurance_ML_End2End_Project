from pydantic import BaseModel, Field
class FeedbackInput(BaseModel):
    text: str = Field(..., min_length=10)
