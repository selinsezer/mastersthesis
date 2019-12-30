import pandas as pd
import numpy as np

keyword = input("Please enter the keyword for the data: ")
path = 'results/' + keyword + '/'
new_file_name = 'corpus/' + keyword + '/' + keyword +  '_annotated.csv'
pos_data_file_name = path + keyword + '.csv'
neg_data_file_name = path + 'non-' + keyword + '.csv'

annotated_df = pd.read_csv(new_file_name, sep=',')
negative_size = len(annotated_df[annotated_df['annotation'] == 0])
positive_size = len(annotated_df[annotated_df['annotation'] == 1])


if positive_size <= negative_size:
    sample_size = positive_size
    positive_data = annotated_df[annotated_df['annotation'] == 1].head(sample_size)['opcode']
    negative_data = annotated_df[annotated_df['annotation'] == 0].head(sample_size)['opcode']

else:
    diff = positive_size - negative_size
    extra_data = pd.read_csv(neg_data_file_name, sep=',').head(diff)
    annotation_results = np.empty(diff, dtype=int)
    annotation_results.fill(0)
    extra_data = extra_data.assign(annotation=annotation_results)
    positive_data = annotated_df[annotated_df['annotation'] == 1]['opcode']
    negative_initial = annotated_df[annotated_df['annotation'] == 0].append(extra_data, sort=True)
    negative_data = negative_initial['opcode']


positive_data.to_csv(pos_data_file_name, index=False, header='opcode')
negative_data.to_csv(neg_data_file_name, index=False, header='opcode')

assert(positive_data.size == negative_data.size)
