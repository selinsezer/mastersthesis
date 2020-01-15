import os
import natsort
from etherscan.contracts import Contract
import json
import pandas as pd
from threading import Thread
import argparse


class Crawler:
    def __init__(self):
        self.resources_path = "resources"
        self.raw_path = os.path.join(self.resources_path, "not_combined")
        self.processed_path = os.path.join(self.resources_path, "processed_not_combined")
        self.data_path = os.path.join(self.resources_path, "data.csv")
        self.key_path = os.path.join(self.resources_path, "api_key.json")
        self.main_data = pd.DataFrame()
        self.learning_path = os.path.join(self.resources_path, "learning_data.csv")
        self.app_path = os.path.join(self.resources_path, "app_data.csv")

        with open(self.key_path, mode='r') as key_file:
            self.key = json.loads(key_file.read())['key']

    def get_resource_code(self, sc_addr):
        api = Contract(address=str(sc_addr), api_key=self.key)
        sourcecode = api.get_sourcecode()
        return sourcecode[0]['SourceCode']

    def process_file(self, fn):
        write_path = os.path.join(self.processed_path, fn)
        fp = os.path.join(self.raw_path, fn)
        df = pd.read_csv(fp, sep=',')

        if os.path.exists(write_path):
            processed_df = pd.read_csv(write_path, sep=',')

            if processed_df.tail(1).values[0][0] == df.tail(1).values[0][0]:
                print("- The file: {}  is already completely processed.".format(fn))
                return

            else:
                last_index_df = df.loc[df['address'] == processed_df.tail(1).values[0][0]].index[0]
                last_index_pdf = processed_df.loc[processed_df['address'] == processed_df.tail(1).values[0][0]].index[0]
                assert(last_index_df == last_index_pdf)
                print("- Continue processing the file: {} with remaining {} data points".format(fn, len(df) - last_index_df))
                df = df[last_index_df+1:]
                i = last_index_df + 1

        else:
            i = 1
            print("- Processing file: {} with {} data points.".format(fn, len(df)))

        processed_df = pd.DataFrame(columns=['address', 'bytecode', 'solidity'])
        for index, row in df.iterrows():
            if i % 500 == 0:
                print("\t{}. Writing mid-results for file: {}.".format(i, fn))
                with open(write_path, 'a') as f:
                    processed_df.to_csv(f, index=False, sep=',', header=f.tell()==0)

                processed_df = pd.DataFrame(columns=['address', 'bytecode', 'solidity'])

            try:
                solidity = self.get_resource_code(row['address'])
                if solidity == '':
                    solidity = 'None'
            except Exception as e:
                print("\t - Cannot obtain solidity for addr: {}. Exception: {}".format(row['address'], e))
                solidity = 'Exception'
            processed_df = processed_df.append(
                [{'address': row['address'], 'bytecode': row['bytecode'], 'solidity': solidity}])
            i += 1

        print("- Writing final results for file: {}.".format(fn))
        with open(write_path, 'a') as f:
            processed_df.to_csv(f, index=False, sep=',', header=False)

    def crawl_resource_code(self, range):
        file_names = os.listdir(self.raw_path)
        file_names = natsort.natsorted(file_names, reverse=False)
        file_names = file_names[range[0]:range[1]]
        threads = list()
        for fn in file_names:
            self.process_file(fn)
            t = Thread(target=self.process_file, args=(fn,))
            threads.append(t)

        for t in threads:
            t.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--range", required=True, type=str, help="The range of files to be processed, separated with - ")
    args = parser.parse_args()
    file_range = args.range.split("-")
    file_range = list(map(int, file_range))
    rcc = Crawler()
    rcc.crawl_resource_code(file_range)
