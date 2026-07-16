from pydantic import BaseModel

class CaptionResponse(BaseModel):
    filename: str
    caption: str
    
class QuestionResponse(BaseModel):
    filename: str
    question: str
    answer: str
    
class SearchMatch(BaseModel):
    filename: str
    similarity: float
    
class SearchResponse(BaseModel):
    matches: list[SearchMatch]