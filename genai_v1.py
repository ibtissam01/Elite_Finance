import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI

# Page title
st.set_page_config(page_title=' Elite Finance')
st.title('EliteGenera Finance')

"""# Gen AI : Exports diversification for Morocco"""

import streamlit as st

# Set the title in blue color and large size
st.markdown('<h1 style="color:blue;">EliteGenera Finance</h1>', unsafe_allow_html=True)

# Set the heading in green color and medium size
st.markdown('<h2 style="color:green;"> #Gen AI: Exports diversification for Morocco</h2>', unsafe_allow_html=True)

from langchain_experimental.agents import create_csv_agent
from langchain.llms import Cohere

agent = create_csv_agent(Cohere(temperature=0, cohere_api_key="sMtMgxOL4fxZtpW0OOFtLFraoAXCsq0FXYwoV0Xi", model='command-nightly'),
                         'ECI_Product_Dataset.csv',
                         verbose=True)

# initialize the callback handler with a container to write to
import streamlit as st

if prompt := st.chat_input("Enter your question"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        response = agent.run(prompt)
        st.write(response)
