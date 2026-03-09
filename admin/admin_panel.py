import streamlit as st
import pandas as pd
import os

st.title("Admin Data Upload")

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

file = st.file_uploader(
    "Upload Business Data",
    type=["csv", "xlsx"]
)

if file:

    try:

        # If Excel file
        if file.name.endswith(".xlsx"):
            df = pd.read_excel(file)

        # If CSV file
        else:
            try:
                df = pd.read_csv(file, encoding="utf-8")
            except:
                df = pd.read_csv(file, encoding="latin1")

        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        path = os.path.join(DATA_FOLDER, file.name)

        df.to_csv(path, index=False)

        st.success("File uploaded and saved successfully")

    except Exception as e:

        st.error(f"Error reading file: {e}")