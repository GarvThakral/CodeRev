from langchain_cohere import ChatCohere
from langchain_community.tools import ReadFileTool 
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools import ListDirectoryTool
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate , PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

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

template = "You are a file editor AI , replace all the text in the given file to 'Hi mom' "

template = ChatPromptTemplate([
    ("system","You are a file editor AI , replace all the text in the given file to 'Hi mom'"),
    ("human","Relace the contents of the file test.py in the current working directory"),
    ("ai","The contents of the file have been replaced t 'Hi mom'"),
    ("human","{input}"),
    ("placeholder", "{agent_scratchpad}")

])

toolkit = FileManagementToolkit()

tools = toolkit.get_tools()

agent = create_tool_calling_agent(llm = model , tools = tools , prompt = template)

agent_executor = AgentExecutor(agent=agent, tools=tools , verbose=True)

print(agent_executor.invoke({"input":"Replace the contents of ./src/klp/tesy.py"}))