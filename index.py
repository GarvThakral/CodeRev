from langchain_cohere import ChatCohere
from langchain_community.tools import ReadFileTool
from langchain_community.tools import ListDirectoryTool
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate , PromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatCohere(temperature=0)

template = ChatPromptTemplate([
        ("system","""You are a code base analyser agent , your job is to find out all the file names present in the given directory strictly using the ListDirectoryTool. Make sure the output is one continuos list of file paths like ... fil1.py' , 'file2.py' , 'src/file3.py"
        Once You are done finding the file names , read through the files strictly using the ReadFileTool . If the file does not contribute to the codebase , consider them useless . Return the file names in structured json .
        """),
        ("human", "What files are in /root/haha"),
        ("ai", "'relevant':'['fil1.py' , 'file2.py' , 'src/file3.py']' , 'irrelevant':'['test.sh','.env']'"),
        ("human", "{prompt}\n\n{agent_scratchpad}"),
])

template = ChatPromptTemplate([
        ("system","You are a code base analyser agent , your job is to find out all the file names present in the given directory strictly using the ListDirectoryTool. Make sure the output is one continuos list of file paths like ... fil1.py' , 'file2.py' , 'src/file3.py'"),
        ("human", "What files are in /root/haha"),
        ("ai", "'relevant':'['fil1.py' , 'file2.py' , 'src/file3.py']' , 'irrelevant':'['test.sh','.env']'"),
        ("human", "{prompt}\n\n{agent_scratchpad}"),
])

tools = [ReadFileTool(),ListDirectoryTool()]

agent = create_tool_calling_agent(model,tools , template)

agent_executor = AgentExecutor(agent=agent, tools=tools , verbose=True)


prompt = PromptTemplate.from_template(
    template = """You are a codebase analyzer AI , given the directory structure , extract files which are relevant to the code base . Meaning keep only file names which contribute to the actual code . For example xyz.py and remove file names which are not important for the codebase , for example test.sh , .env"""
)

chain = agent_executor | model | prompt

print(chain.invoke({"prompt":"List all files in the current directory and its sub directories "})['output'])
