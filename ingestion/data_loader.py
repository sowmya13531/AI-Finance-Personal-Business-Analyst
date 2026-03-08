import pandas as pd
import os

DATA_FOLDER = "data"

def load_data():

    data = {}

    if not os.path.exists(DATA_FOLDER):
        return data

    for file in os.listdir(DATA_FOLDER):

        if file.endswith(".csv"):

            path = os.path.join(DATA_FOLDER, file)

            try:
                df = pd.read_csv(path)

                key = file.replace(".csv","")

                data[key] = df

            except:
                pass

    return data