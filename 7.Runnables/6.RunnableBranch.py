from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough,RunnableBranch

# Load env
load_dotenv()

def word_count(text):
    return len(text.split(" ")) > 300


huggingFaceToken = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=huggingFaceToken,
    temperature=0.0
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# Prompts
prompt1 = PromptTemplate(
    template="Write a detailed report about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Write a summary report about {topic}",
    input_variables=["topic"]
)

report_gen_chain=RunnableSequence(prompt1,model,parser)
branch_chain=RunnableBranch(
    (word_count,RunnableSequence(prompt2,model,parser)) ,
    RunnablePassthrough()

)
final_chain=RunnableSequence(report_gen_chain,branch_chain)
print(final_chain.invoke({"topic":"paris"}))