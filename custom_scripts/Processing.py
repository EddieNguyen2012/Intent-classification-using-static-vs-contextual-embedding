import os
import json
import pandas as pd

class DataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_df = None
        self.test_df = None
        self.val_df = None
        self.create_dataframes()

    def load_data(self):
        with open(os.path.join(self.data_dir, "data_full.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def create_dataframes(self):
        data = self.load_data()
        train_df = pd.DataFrame(data["train"], columns=["text", "intent"])
        val_df   = pd.DataFrame(data["val"], columns=["text", "intent"])
        test_df  = pd.DataFrame(data["test"], columns=["text", "intent"])
        oos_train_df = pd.DataFrame(data["oos_train"], columns=["text", "intent"])
        oos_test_df = pd.DataFrame(data["oos_test"], columns=["text", "intent"])
        oos_val_df   = pd.DataFrame(data["oos_val"], columns=["text", "intent"])

        self.train = pd.concat([train_df, oos_train_df], ignore_index=True)
        self.test = pd.concat([test_df, oos_test_df], ignore_index=True)
        self.val = pd.concat([val_df, oos_val_df], ignore_index=True)


    def get_train_df(self):
        return self.train


    def get_test_df(self):
        return self.test


    def get_val_df(self):
        return self.val

