from lambdamart import LambdaMART
import numpy as np
import pandas as pd
import json
import train


model = LambdaMART()
model.load('model/lambdamart_model_30.lmart')

train_path=["example_data/part-00000"]
test_path=["example_data/part-00001"]
train=train.get_data(train_path)
print(train[0])
print(train[1])
print(train[2])

print(model.predict(train[:100,1:]))