from langchain_cohere import ChatCohere
from langchain_community.tools import ReadFileTool , tool
from langchain_community.agent_toolkits import FileManagementToolkit  
from langchain_community.tools import BearlyInterpreterTool
from langchain_community.tools import ListDirectoryTool
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate , PromptTemplate , MessagesPlaceholder
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import sys
from typing import Annotated 
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser

from langchain_openai import ChatOpenAI
import json
load_dotenv()

# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
# model = ChatCohere(temperature=0)
OPENROUTER_API_KEY = "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"


model = ChatOpenAI(
    model="x-ai/grok-4-fast:free",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model_kwargs={"temperature": 0}   # instead of client_kwargs
)
template = ChatPromptTemplate([
    ("system", """You are a codebase analyzer. Use ListDirectoryTool to find all files in directories and subdirectories. Use ReadFileTool to examine file contents. Categorize files as relevant (code files) or irrelevant (config/system files) . Make sure the file names contain the relative path for the files , include all code files . Also return the output in structured json format without any trailing text: 
    {{
     "relevant": ["file1.py , src/klp/tesy.py"],
      "irrelevant": [".env"]}}
     """
     ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

tools = [ReadFileTool(),ListDirectoryTool()]

agent = create_tool_calling_agent(model,tools , template)

agent_executor = AgentExecutor(agent=agent, tools=tools , verbose=True)

output = agent_executor.invoke({"input": "List all files in the current directory and its sub directories"})['output']
s = output.strip()
if s.startswith("```"):
    lines = s.splitlines()
    s = "\n".join(lines[1:-1])

data = json.loads(s)
relevantList = data['relevant']
relevantListFiltered = [x for x in relevantList if x != "index.py"]
print(relevantListFiltered)

template = ChatPromptTemplate([
    ("system","""You are a code optimizer AI , read all the text given in the file index.py and if you see any scope of optimization for example , nested loops , vectorized implementation or being effficient in any sense . Try to keep majority of the original implementation , try to optimize the code as little as possible. Also look for any missing imports , variable mismatchs , memory leaks , vulnerabilities , API_KEYS and then suggest the changes by creating another file called 'original_file_name'_replace.py .
     Do not escape them into HTML entities like &lt;, &gt;, &amp;. 
        For example:
        - Write "<tag>" instead of "&lt;tag&gt;"
        - Write "&" instead of "&amp;"
     Make sure the new file is created for the user to see it . Also return the output in structured json format like this : 
    {{
        "original":"print(1+12)",
     "optimized":"print(13)"
     }}

     Make sure it is a proper json string and not something like ```json....``` .
     """),
    ("human","Try to optimize ./src/klp/tesy.py"),
    ("ai","""
    {{
        "original":"print(1+12)",
     "optimized":"print(13)"
     }}
        """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{input}"),
    ("placeholder", "{agent_scratchpad}")

])

toolkit = FileManagementToolkit()

tools = toolkit.get_tools()

class ExpectedOptimizationOutput(BaseModel):
    original:str
    optimized:str

parser = PydanticOutputParser(pydantic_object=ExpectedOptimizationOutput)

agent = create_tool_calling_agent(llm = model , tools = tools , prompt = template)

memory = [
]

agent_executor = AgentExecutor(agent=agent , tools=tools , verbose=True , output_parser=parser)

for x in relevantListFiltered:
    json_output = agent_executor.invoke({"input":f"Optimize the following code file {x}","chat_history":memory})["output"]
    memory.append(AIMessage(json_output))   
    json_output = json.loads(json_output)
    human = input("accept or deny or re-eval")
    memory.append(HumanMessage(human))
    if(human == "accept"):
        json_output = agent_executor.invoke({"input":f"The changes to the file {x} were accepted , please make the changes to the {x} and remove the replace file created earlier","chat_history":memory})["output"]
        memory.append(AIMessage(json_output))
        continue
    elif human == "deny":
        json_output = agent_executor.invoke({"input":f"The changes to the file {x} were rejected , keep the original file {x} and remove the replace file created earlier","chat_history":memory})["output"]
        memory.append(AIMessage(json_output))
        continue
    elif human == "re-eval":
        while human not in ["accept", "deny"]:
            json_output = agent_executor.invoke({"input":f"Optimize the following file {x}","chat_history":memory})["output"]
            memory.append(AIMessage(json_output))
            human = input("accept or deny or re-eval")
            memory.append(HumanMessage(human))
            if(human == "accept"):
                json_output = agent_executor.invoke({"input":f"The changes to the file {x} were accepted , please make the changes to the original file {x} and remove the replace file created earlier","chat_history":memory})["output"]
                memory.append(AIMessage(json_output))
                break
            elif human == "re-eval":
                continue
            else:
                json_output = agent_executor.invoke({"input":f"The changes to the file {x} were rejected , keep the original file {x} and remove the replace file created earlier","chat_history":memory})["output"]
                memory.append(AIMessage(json_output))
                break



sys.exit()

print(json_output["original"])
print(json_output["optimized"])

@tool
def test_run_tool(input:str):
    """
    This is a custom tool for the test runner agent . The tests are supposed to be written in a file with the suitable extension (for example : .py , .java , .cpp ,etc) . 
    This tool expects as input the command to run the test file in this particular format
    'python test_execution_time.py'
    This file returns the output of the subprocess from the CLI.
    """
    import subprocess
    result = subprocess.run(input.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}


test_writer_tools = [test_run_tool] + tools


test_writer_template = ChatPromptTemplate([
    ("system","""You are a test writer agent for optimized code testing . Given the original code , and the optimized code write a test in the necessary language , to compare the execution times of the original and the optimized code . Add neccessary logging to later compare the run times , proceed to then run the test in the working directory and extract the times taken for both the test to run and output the comparision . """),
    ("human","{input}"),
    ("placeholder","{agent_scratchpad}")
    # ("ai","The tests have been created in ./test_runid.py . The results (4 seconds vs 1 second) show a 3 second decrease . "),
]) 

test_writer_agent = create_tool_calling_agent(llm = model , tools = test_writer_tools , prompt = test_writer_template)

test_writer_agent_exeutor = AgentExecutor(agent = test_writer_agent , tools = test_writer_tools , verbose = True)

print(test_writer_agent_exeutor.invoke({"input": json.dumps(json_output)}))

