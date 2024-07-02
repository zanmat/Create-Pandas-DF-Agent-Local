from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama


import pandas as pd
from langchain_openai import OpenAI


df = pd.read_csv(
   "https://raw.githubusercontent.com/zanmat/Create-Pandas-DF-Agent-Local/main/data/data.csv"
)


#agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True, allow_dangerous_code=True)


agent = create_pandas_dataframe_agent(
   ChatOllama(temperature=0, model="mistral"),
   df,
   verbose=True,
#    agent_type=AgentType.OPENAI_FUNCTIONS,
   allow_dangerous_code=True
)


agent.invoke("how many refugees were there in Peru in 2023?")


