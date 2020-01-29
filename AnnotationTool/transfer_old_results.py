import pandas as pd
import numpy as np
import os

keyword = input('Please enter the keyword to work on: ')
new_file_name = 'corpus/' + keyword + '/' + keyword + '.csv'
old_annotated_data_path = 'corpus/' + keyword + '/' + keyword + '_annotated.csv'

if os.path.exists(old_annotated_data_path):
    old_annotated_data = pd.read_csv(old_annotated_data_path, sep=',')
    new_data = pd.read_csv(new_file_name, sep=',')
    new_annotation_results = np.empty(new_data['solidity'].size, dtype=int)
    new_annotation_results.fill(-1)
    new_data = new_data.assign(annotation=new_annotation_results)
    odd_cases = 0
    initial_length = len(new_data)
    for index, row in old_annotated_data.iterrows():
        if len(new_data[new_data['address'] == row['address']]) == 0:
            if row['annotation'] == 1:
                odd_cases += 1
                to_add = old_annotated_data.loc[old_annotated_data['address'] == row['address']]
                new_data = new_data.append(to_add)
                new_data.loc[new_data['address'] == row['address'], 'annotation'] = -1
        else:
            new_data.loc[new_data['address'] == row['address'], 'annotation'] = row['annotation']
    final_length = len(new_data)
    assert(final_length == initial_length + odd_cases)
    new_data.to_csv(old_annotated_data_path, sep=',', index=False)
    # print(len(old_annotated_data))

