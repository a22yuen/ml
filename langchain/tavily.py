from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

search = TavilySearchResults()

print(search.invoke("what is the weather in SF"))

