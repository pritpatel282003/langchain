from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

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
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# Prompts
joke_prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

explain_prompt = PromptTemplate(
    template="Explain this joke: {joke}",
    input_variables=["joke"]
)

# Chains
joke_gen_chain = RunnableSequence(joke_prompt, model, parser)
explain_chain = RunnableSequence(explain_prompt, model, parser)

# Final chain:
# 1. Generate joke
# 2. Pass it through RunnablePassthrough
# 3. In parallel: keep the joke + explain it
final_chain = RunnableSequence(
    joke_gen_chain,
    RunnableParallel({
        "joke": RunnablePassthrough(),   # just forward the joke
        "explanation": explain_chain     # send the joke into explanation
    })
)

# Run it
result = final_chain.invoke({"topic": "man"})
print(result)
