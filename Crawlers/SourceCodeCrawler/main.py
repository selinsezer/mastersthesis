import os
import natsort
from etherscan.contracts import Contract
import json
import pandas as pd
from threading import Thread


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
        processed_df = pd.DataFrame(columns=['address', 'bytecode', 'solidity'])
        write_path = os.path.join(self.processed_path, fn)
        fp = os.path.join(self.raw_path, fn)
        df = pd.read_csv(fp, sep=',')
        i = 1
        print("- Processing file: {} with {} data points.".format(fn, len(df)))
        for index, row in df.iterrows():
            try:
                solidity = self.get_resource_code(row['address'])
                if solidity == '':
                    solidity = 'None'
            except Exception as e:
                print("\t - Cannot obtain solidity for addr: {}. Exception: {}".format(row['address'], e))
                solidity = 'Exception'
            processed_df = processed_df.append(
                [{'address': row['address'], 'bytecode': row['bytecode'], 'solidity': solidity}])
            if i % 5000 == 0:
                print("\t{}. Writing mid-results for file: {}.".format(i, fn))
                processed_df.to_csv(write_path, index=False, sep=',')
            i += 1
        print("- Writing final results for file: {}.".format(fn))
        processed_df.to_csv(write_path, index=False, sep=',')

    def crawl_resource_code(self):
        file_names = os.listdir(self.raw_path)
        file_names = natsort.natsorted(file_names, reverse=False)
        file_names = file_names[1:5]
        threads = list()
        for fn in file_names:
            t = Thread(target=self.process_file, args=(fn,))
            threads.append(t)

        for t in threads:
            t.start()


if __name__ == "__main__":
    rcc = Crawler()
    rcc.crawl_resource_code()
