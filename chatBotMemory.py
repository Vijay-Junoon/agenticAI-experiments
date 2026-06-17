from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from langchain.messages import HumanMessage,AIMessage
from typing import TypedDict,List,Union
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model='qwen/qwen3-32b',reasoning_format="hidden")

class AgentState(TypedDict):
  messages: List[Union[HumanMessage,AIMessage]]

def process(state: AgentState) -> AgentState:
  response = llm.invoke(state['messages'])
  state['messages'].append(AIMessage(content = response.content))
  print(f"AI: {response.content}")
  return state

graph = StateGraph(AgentState)
graph.add_node('process',process)
graph.add_edge(START,'process')
graph.add_edge('process',END)
app = graph.compile()

conversation_history = []
user_input = ""
prompt = "(Answer clearly. Do not include any sort of emojis.)"
while user_input != 'exit':
  user_input = input("User: ")
  conversation_history.append(HumanMessage(user_input + prompt))
  response = app.invoke({'messages': conversation_history})
  conversation_history = response['messages']


with open("history.txt", "w") as history:
  for message in conversation_history:
    if isinstance(message,HumanMessage):
      txt = "User: " + message.content
      history.write(f"{txt} \n")
    else:
      txt = "AI: "  + message.content
      history.write(f"{txt} \n \n")

  history.write("End Of Converstation \n\n")

print(conversation_history)