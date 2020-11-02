from parsers import MakeModel
import os

ticker = "AMZN"

model = MakeModel(ticker, os.getcwd())
model.get_data()
# model.train_model()
# model.test_model()


