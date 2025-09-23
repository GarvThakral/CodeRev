from langchain_cohere import ChatCohere
from langchain_community.tools import ReadFileTool , tool
from langchain_community.agent_toolkits import FileManagementToolkit  
from langchain_community.tools import BearlyInterpreterTool
from langchain_community.tools import ListDirectoryTool
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate , PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json
load_dotenv()

# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
model = ChatCohere(temperature=0)


# template = ChatPromptTemplate([
#     ("system", """You are a codebase analyzer. Use ListDirectoryTool to find all files in directories and subdirectories. Use ReadFileTool to examine file contents. Categorize files as relevant (code files) or irrelevant (config/system files). Output final JSON: {{"relevant": ["file1.py"], "irrelevant": [".env"]}}"""),
#     ("human", "{input}"),
#     ("placeholder", "{agent_scratchpad}")
# ])

# tools = [ReadFileTool(),ListDirectoryTool()]

# agent = create_tool_calling_agent(model,tools , template)

# agent_executor = AgentExecutor(agent=agent, tools=tools , verbose=True)

# output = agent_executor.invoke({"input": "List all files in the current directory and its sub directories"})['output']

# print(output)
# print(json.loads(output.replace('`','')[4:])['relevant'])


template = ChatPromptTemplate([
    ("system","""You are a code optimizer AI , read all the text given in the file index.py and if you see any scope of optimization for example , nested loops , vectorized implementation or being effficient in any sense , then suggest the changes in another file called 'original_file_name'_replace.py . Also return the output in structured json format like this : 
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
    ("human","{input}"),
    ("placeholder", "{agent_scratchpad}")

])

toolkit = FileManagementToolkit()

tools = toolkit.get_tools()

agent = create_tool_calling_agent(llm = model , tools = tools , prompt = template)

agent_executor = AgentExecutor(agent=agent, tools=tools , verbose=True)

json_output = agent_executor.invoke({"input":"Replace the contents of ./src/klp/tesy.py"})["output"]

print("HEre")
print(json_output)
print("HEre")
json_output = json.loads(json_output)

print(json_output["original"])
print(json_output["optimized"])

@tool
def test_run_tool(input):
    """
    This is a custom tool for the test runner agent . The tests are supposed to be written in a file with the suitable extension (for example : .py , .java , .cpp ,etc) . 
    This tool expects as input the command to run the test file in this particular format
    ["python","test_execution_time.py"].
    This file returns the output of the subprocess from the CLI.
    """
    import subprocess
    result = subprocess.run(['ls', '-l'], stdout=subprocess.PIPE)
    result = subprocess.run(input, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}


test_writer_tools = [test_run_tool] + tools


test_writer_template = ChatPromptTemplate([
    ("system","""You are a test writer agent for optimized code testing . Given the original code , and the optimized code write a test in the necessary language , to compare the execution times of the original and the optimized code . Add neccessary logging to later compare the run times , proceed to then run the test in the working directory and extract the times taken for both the test to run and output the comparision . """),
    ("human","{input}"),
    ("placeholder","{agent_scratchpad}")
    # ("ai","The tests have been created in ./test_runid.py . The results (4 seconds vs 1 second) show a 3 second decrease . "),
]) 

test_writer_agent = create_tool_calling_agent(llm = model , tools = tools , prompt = test_writer_template)

test_writer_agent_exeutor = AgentExecutor(agent = test_writer_agent , tools = test_writer_tools , verbose = True)

print(test_writer_agent_exeutor.invoke({"input": json.dumps(json_output)})
)
