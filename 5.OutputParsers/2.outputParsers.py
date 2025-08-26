'''from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
load_dotenv()
huggingFaceToken=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=huggingFaceToken
)
model=ChatHuggingFace(llm=llm)

#1st prompt -> Detailed Response
template1=PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

template2=PromptTemplate(
    template="Write a 5 line summary on {text}",
    input_variables=['text']
)

prompt1=template1.invoke({"topic":"blackhole"})
result=model.invoke(prompt1)
prompt2=template2.invoke({"text":result.content})
result1=model.invoke(prompt2)
print(result1.content)'''


'''
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # or "gemini-2.5-pro"
    google_api_key=api_key,
    temperature=0.7
)

# Prompt 1 → Detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# Prompt 2 → Summary
template2 = PromptTemplate(
    template="Write a 5 line summary on {text}",
    input_variables=["text"]
)

# Step 1: Generate detailed report
prompt1 = template1.invoke({"topic": "blackhole"})
result = model.invoke(prompt1)

# Step 2: Generate summary from detailed report
prompt2 = template2.invoke({"text": result.content})
result1 = model.invoke(prompt2)

print("=== Detailed Report ===")
print(result.content)
print("\n=== Summary ===")
print(result1.content)''''''

'''

'''from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # or "gemini-2.5-pro"
    google_api_key=api_key,
    temperature=0.7
)

# Prompt 1 → Detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# Prompt 2 → Summary
template2 = PromptTemplate(
    template="Write a 5 line summary on {text}",
    input_variables=["text"]
)

parser=StrOutputParser()
chain=template1 | model | parser | template2 | model | parser

result=chain.invoke({"topic":"big bang theory"})
print(result)'''

# JSON Parser
'''from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()
huggingFaceToken=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=huggingFaceToken
)
model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()
#1st prompt -> Detailed Response
template1=PromptTemplate(
    template="Give me the batman , age and city of the fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={
        "format_instruction":parser.get_format_instructions()
    }
)

chain=template1 | model | parser
result=chain.invoke({})
print(result)'''

'''from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()
huggingFaceToken = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=huggingFaceToken
)
model = ChatHuggingFace(llm=llm)

# Define schema
schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

# Prompt includes parser instructions separately
template1 = PromptTemplate(
    template=(
        "Give me 3 facts about {topic}.\n\n"
        "Format the response as JSON below:\n{format_instructions}"
    ),
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Build chain
chain = template1 | model | parser

# Run
result = chain.invoke({"topic": "Clark Kent"})
print(result)'''

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load env
load_dotenv()
huggingFaceToken = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Define schema
class PersonInfo(BaseModel):
    name: str = Field(description="The full name of the person")
    age: int = Field(description="The age of the person in years")
    city: str = Field(description="The city where the person lives")

# Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=huggingFaceToken,
    temperature=0.0
)
model = ChatHuggingFace(llm=llm)

# Output parser
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# Prompt with format instructions
template = PromptTemplate(
    template=(
        "You are a strict JSON generator.\n"
        "Extract details of the following person strictly in JSON format.\n\n"
        "{format_instructions}"
    ),
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Chain
chain = template | model | parser

# Example run
result = chain.invoke({})
print(result)





