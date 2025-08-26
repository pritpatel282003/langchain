import random

class FakeLLM1:
    def __init__(self):
        print("LLM created")

    def predict(self,prompt):
        response_list=[
            'Delhi is the capital of india',
            'IPL is a cricket League',
            'AI stands for Artificial Intelligence'
        ]
        return {
            'response':random.choice(response_list)
        }
    
class FakeLLM2:
    def __init__(self,template,input_variables):
        self.tempplate=template
        self.input_variables=input_variables

    def format(self,input_dict):
        return self.tempplate.format(**input_dict)
    
class FakeChain:
    def __init__(self,llm,prompt):
        self.llm=llm
        self.prompt=prompt
    def run(self,input_dict):
        final_prompt=self.prompt.format(input_dict)
        self.llm.predict(final_prompt)
        
    
llm=FakeLLM1()
result=llm.predict('What is the capital of india')
print(result)

template=FakeLLM2(
    template='Write a {length} poem about {topic}',
    input_variables=['length','topic']
)

llm=FakeLLM1()
result1=template.format({'length':'short','topic':'india'})
print(result1)
llm.predict(result1 )