import streamlit as st
import pandas as pd

st.title("Admin Data Upload")

file = st.file_uploader("Upload Business CSV")

if file:

    df = pd.read_csv(file)

    path = "data/" + file.name

    df.to_csv(path,index=False)

    st.success("File uploaded successfully")