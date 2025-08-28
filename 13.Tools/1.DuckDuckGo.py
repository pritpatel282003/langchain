from langchain_community.tools import DuckDuckGoSearchRun
search_tool=DuckDuckGoSearchRun()
result=search_tool.invoke('SA VS AUS cricket series')
print(result)