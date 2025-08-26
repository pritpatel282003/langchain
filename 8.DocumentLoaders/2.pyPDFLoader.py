from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field

# Load env
load_dotenv()
huggingFaceToken = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# ==========================
# Define output schema
# ==========================
class Rating(BaseModel):
    rating: int = Field(..., description="Rating out of 10")

# Pydantic parser
parser = PydanticOutputParser(pydantic_object=Rating)

# Output fixing parser (fallback if model messes up)
llm_raw = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=huggingFaceToken,
    temperature=0.0
)
model = ChatHuggingFace(llm=llm_raw)

rating_parser = OutputFixingParser.from_llm(parser=parser, llm=model)

# ==========================
# Prompts
# ==========================
summary_prompt = PromptTemplate(
    template="Write a summary in one paragraph: {topic}",
    input_variables=["topic"]
)

rating_prompt = PromptTemplate(
    template=(
        "Rate him on a scale of 10 as a AIML developer.\n\n"
        "Summary: {summary}\n\n"
        "Return ONLY in JSON format as below:\n{format_instructions}"
    ),
    input_variables=["summary"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# ==========================
# Load PDF
# ==========================
loader = PyPDFLoader("Prit Sanjiv Patel_AIML.pdf")
docs = loader.load()

# ==========================
# Chains
# ==========================
summary_chain = summary_prompt | model 
parallel_chain = RunnableParallel({
    "summary": RunnablePassthrough(),
    "rate": rating_prompt | model | rating_parser
})

final_chain = summary_chain | parallel_chain

# ==========================
# Run on first page of PDF
# ==========================
result = final_chain.invoke({"topic": docs[0].page_content})
print(result)
