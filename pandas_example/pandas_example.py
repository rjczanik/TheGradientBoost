import pandas as pd

train_df = pd.read_csv('data/train.csv')
binary_indicator = train_df['Survived']

print(train_df)
print(binary_indicator)

samp_dict = {"A": 2, "B": 4}
damp_df = pd.DataFrame.from_dict(samp_dict, orient='index')

survivors_df = train_df[train_df['Survived'] == 1]

train_df['Age'] = 18
