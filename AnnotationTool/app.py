from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import os
from difflib import SequenceMatcher

app = Flask(__name__)
keyword = input('Please enter the keyword to work on: ')
file_name = 'corpus/' + keyword + '/' + keyword + '_not_annotated.csv'
new_file_name = 'corpus/' + keyword + '/' + keyword + '_annotated.csv'

if not os.path.exists(new_file_name):
    df = pd.read_csv(file_name, sep=',')
    annotation_results = np.empty(df['solidity'].size, dtype=int)
    annotation_results.fill(-1)
    annotated_df = df.assign(annotation=annotation_results)
    annotated_df.to_csv(new_file_name, sep=',', index=False)
else:
    annotated_df = pd.read_csv(new_file_name, sep=',')


def get_current_status():
    not_annotated = annotated_df[annotated_df['annotation'] == -1]['solidity'].size
    negative = annotated_df[annotated_df['annotation'] == 0]['solidity'].size
    positive = annotated_df[annotated_df['annotation'] == 1]['solidity'].size
    print("Number of not annotated contracts: {}".format(not_annotated))
    print("Number of pos contracts: {}".format(positive))
    print("Number of neg contracts: {}".format(negative))


def get_match_ratio(sc, to_compare):
    s = SequenceMatcher(lambda x: x == " ", sc, to_compare)
    if s.ratio() >= 0.6:
        print("Found a similar({}) contract, labeling it for you...".format(s.ratio()))
        return True
    return False


def check_for_function(keyword, value):
    for index, row in annotated_df[annotated_df['annotation'] == -1].iterrows():
        if keyword in row['solidity']:
            annotated_df.loc[annotated_df['address'] == row['address'], 'annotation'] = value
    annotated_df.to_csv(new_file_name, sep=',', index=False)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form['submit_button'] == 'Positive':
            value = 1
        else:
            value = 0

        #todo find an efficient check here..
        source_code = annotated_df.loc[annotated_df['address'] == request.form['addr']]['solidity'].values[0]
        functions = annotated_df.loc[annotated_df['address'] == request.form['addr']]['function_names'].values[0]
        annotated_df.loc[annotated_df['function_names'] == functions, ['annotation']] = value

        counter = 0
        for index, row in annotated_df[annotated_df['annotation'] == -1].iterrows():
            counter += 1
            if counter > 5:
                break
            if get_match_ratio(source_code, row['solidity']):
                counter = 0
                annotated_df.loc[annotated_df['address'] == row['address'], 'annotation'] = value
                functions = annotated_df.loc[annotated_df['address'] == row['address']]['function_names'].values[0]
                annotated_df.loc[annotated_df['function_names'] == functions, ['annotation']] = value
                annotated_df.to_csv(new_file_name, sep=',', index=False)

        annotated_df.loc[annotated_df['address'] == request.form['addr'], 'annotation'] = value
        annotated_df.to_csv(new_file_name, sep=',', index=False)
        next_row = annotated_df[annotated_df['annotation'] == -1].head(1)
        get_current_status()
        if next_row.empty is True:
            return render_template('index.html', contract="Annotation complete.", addr="")
        else:
            return render_template('index.html', contract=next_row['solidity'].values[0], addr=next_row['address'].values[0])

    else:
        next_row = annotated_df[annotated_df['annotation'] == -1].head(1)
        get_current_status()
        return render_template('index.html', contract=next_row['solidity'].values[0], addr=next_row['address'].values[0])


if __name__ == "__main__":
    # check_for_function("function voteFor()  returns (bool success){", 1)
    app.run()
