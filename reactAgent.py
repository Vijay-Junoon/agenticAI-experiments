from typing import Annotated,Sequence,TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph,END
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from IPython import display
from IPython.display import Image

load_dotenv()




class AgentState(TypedDict):
  messages: Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a: int,b: int) -> int:
  """Used to add two given numbers."""
  return a + b

@tool
def subtract(a: int,b: int) -> int:
  """Used to subtract two given numbers."""
  return a - b
tools = [add,subtract]
model = ChatGroq(model='qwen/qwen3-32b',reasoning_format='hidden').bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
  system_prompt = SystemMessage("You are my AI Agent.")

  response = model.invoke(state['messages'] + [system_prompt])
  return {'messages':[response]}

def should_continue(state: AgentState):
  messages = state['messages']
  if not messages[-1].tool_calls:
    return 'exit'
  return 'continue'


graph = StateGraph(AgentState)
graph.add_node('agent',model_call)

tool_node = ToolNode(tools = tools)
graph.add_node('tools',tool_node)

graph.set_entry_point('agent')
graph.add_conditional_edges(
  'agent',
  should_continue,
  {
    'exit': END,
    'continue': 'tools'
  }
)


graph.add_edge('tools','agent')
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 and then subract the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))