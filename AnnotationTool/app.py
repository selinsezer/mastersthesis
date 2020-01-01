from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
keyword = input('Please enter the keyword to work on: ')
file_name = 'corpus/' + keyword + '/' + keyword + '_not_annotated.csv'
df = pd.read_csv(file_name, sep=',')
new_file_name = 'corpus/' + keyword + '/' + keyword + '_annotated.csv'

if not os.path.exists(new_file_name):
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


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form['submit_button'] == 'Positive':
            value = 1
        else:
            value = 0
        annotated_df.loc[annotated_df['address'] == request.form['addr'], ['annotation']] = value
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
    app.run()
