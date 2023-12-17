# -*- coding: utf-8 -*-
"""GenAI_V1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xaoD81_QRUlR3XobD2G8UNFPsbwkjWvY

# EDA
"""
import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI

# Page title
st.set_page_config(page_title='🦜🔗 Elite Finance')
st.title('🦜🔗Elite Finance')

"""# Gen AI : Elite Finance"""



from langchain_experimental.agents import create_csv_agent
from langchain.llms import Cohere


agent = create_csv_agent(Cohere(temperature=0, cohere_api_key="sMtMgxOL4fxZtpW0OOFtLFraoAXCsq0FXYwoV0Xi", model = 'command-nightly'),
                         'ECI_Product_Dataset.csv',
                         verbose=True)


#agent.cohere.llm_chain.prompt.template

# initialize the callback handler with a container to write to
import streamlit as st

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        response = agent.run(prompt)
        st.write(response)
