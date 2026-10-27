# 허승표

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel
from rag.chatbot import rag_answer

app = FastAPI(title="RAG ChatBot")


class ChatRequest(BaseModel):
    question: str
    user_type: str


@app.post("/chat")
def chat(data: ChatRequest):
    result = rag_answer(
        question = data.question,
        user_type = data.user_type.upper() 
    )
    return result
