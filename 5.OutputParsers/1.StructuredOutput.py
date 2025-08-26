from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

class Review(BaseModel):
    summary: str = Field(..., description="A summary of the review in one line")
    sentiment: str = Field(..., description="Sentiment of the review (Positive, Negative, or Neutral)")

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0
)

structured_llm = llm.with_structured_output(Review)

review_text = "I got promoted at Google"

result: Review = structured_llm.invoke(
    f"Please analyze the following review:\n\n{review_text}"
)

print(result.dict())       # {'summary': '...', 'sentiment': 'Positive'}
print(result.sentiment)
