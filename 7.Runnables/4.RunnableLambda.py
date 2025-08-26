from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda,RunnableParallel,RunnablePassthrough,RunnableSequence

# Load env
load_dotenv()

def word_count(text):
    return len(text.split(' '))

huggingFaceToken = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=huggingFaceToken,
    temperature=0.0
)
prompt=PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)




model=ChatHuggingFace(llm=llm)
parser=StrOutputParser()

joke_gen_chain=RunnableSequence(prompt,model,parser)
parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_count)
})

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)
print(final_chain.invoke({"topic":"man"}))
