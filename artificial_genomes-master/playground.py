import pandas as pd
input = pd.read_csv("f8_chr13_brca2_copd_hmb.hapt",sep = ' ',header=None)
print(input.head(5))
input = input.sample(frac=1).reset_index(drop=True)
print(input.head(5))
i = 0
dropped = []
while i < 500:
        if (input[i] == 0).all():
               	print('found',i)
                dropped.append(i)
                input = input.drop(columns=[i])
        i+=1
print(input.shape)            
print(dropped)
print(len(dropped))
