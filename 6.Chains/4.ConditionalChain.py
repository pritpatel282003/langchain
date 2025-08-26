from typing import Literal
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel, Field
from langchain.schema.runnable import Runnable,RunnableBranch,RunnableLambda

# Load env
load_dotenv()
huggingFaceToken = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=huggingFaceToken,
    temperature=0.0
)

# Define structured schema
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description="Sentiment of the feedback (positive or negative)"
    )

# Parser for schema
parser=StrOutputParser()
parser1 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt with format instructions
prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback text into "
        "positive or negative:\n{feedback}\n{format_instructions}"
    ),
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser1.get_format_instructions()}
)

# Model
model = ChatHuggingFace(llm=llm)

# Chain with parser1
classifier_chain = prompt1 | model | parser1

prompt2=PromptTemplate(
    template="Write an appropiate response of the positive feedback : {feedback}",
    input_variables=['feedback']
)
prompt3=PromptTemplate(
    template="Write an appropiate response of the negative feedback : {feedback}",
    input_variables=['feedback']
)

branch_chain=RunnableBranch(
    (lambda x:x['sentiment']=='positive',prompt2 | model | parser),
    (lambda x:x['sentiment']=='negative',prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)
chain= classifier_chain | branch_chain
print(chain.invoke({"feedback":"heyyyyyyya"}))
