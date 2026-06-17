from typing import TypedDict,List
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from langchain.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model='qwen/qwen3-32b')
class AgentState(TypedDict):
  message: List[HumanMessage]
  output: str
def chat(state: AgentState) -> AgentState:
  state['output'] = llm.invoke(state['message']).content
  print(f"Bot: {state['output']}")
  return state

graph = StateGraph(AgentState)
graph.add_node('chat',chat)
graph.add_edge(START,'chat')
graph.add_edge('chat',END)
app = graph.compile()

user_input = input("Enter:")
app.invoke({'message': [HumanMessage(user_input)]})




