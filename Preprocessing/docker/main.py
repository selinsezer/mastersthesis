import pandas as pd
from evmdasm import EvmBytecode
# from threading import Thread
import os
import natsort
import json
from etherscan.contracts import Contract


class Preprocessor:
    def __init__(self):
        self.resources_path = "resources"
        self.raw_path = os.path.join(self.resources_path, "finished_crawling")
        self.processed_path = os.path.join(self.resources_path, "preprocessed")
        self.exceptions_fixed_path = os.path.join(self.resources_path, "exceptions_fixed")
        self.dataset_path = os.path.join(self.resources_path, "dataset")
        self.key_path = os.path.join(self.resources_path, "api_key.json")
        with open(self.key_path, mode='r') as key_file:
            self.key = json.loads(key_file.read())['key']

    def byte2opcode(self, evm_bytecode):
        evmcode = EvmBytecode(evm_bytecode)  # can be hexstr, 0xhexstr or bytes
        evminstructions = evmcode.disassemble()
        opcode = " ".join([str(ei) for ei in evminstructions])
        return opcode

    def process_file(self, fn):
        print("- Processing file: {}".format(fn))
        print("\tReading all smart contracts.")
        fp = os.path.join(self.raw_path, fn)
        df = pd.read_csv(fp, sep=',')
        initial = df.shape[0]
        print("\t-------------------------------------------------")
        print("\tInitial number of contracts: {}".format(initial))
        df = df.drop_duplicates(subset=['bytecode'], keep='first')
        later = df.shape[0]
        print("\tNumber of contracts without the duplicate bytecodes: {}".format(later))
        print("\tNumber of contracts removed: {}".format(initial - later))
        print("\t-------------------------------------------------")
        print("\tTurning bytecodes to opcodes.")
        opcodes = []
        for bytecode in df['bytecode']:
            try:
                opcode = self.byte2opcode(bytecode)
            except:
                opcode = ""
            opcodes.append(opcode)
        df = df.assign(opcode=opcodes)
        print("\t-------------------------------------------------")
        print("#\tRemoving corrupted bytecode.")
        initial = df.shape[0]
        df = df.drop(df[df.opcode == ''].index)
        later = df.shape[0]
        print("{} smart contracts are removed. {} contracts remain.".format(initial - later, df.shape[0]))
        print("\t-------------------------------------------------")
        print("#\tCleaning the opcodes  from arguments and invalid opcodes.")
        clean_opcodes = []
        invalid_opcodes = ["unknown", "invalid", "unofficial", "extcodehash", "pc"]
        for opcode in df['opcode']:
            lines_removed = opcode.split(" ")
            args_removed = [opcode for opcode in lines_removed if "0x" not in opcode]
            for invalid in invalid_opcodes:
                args_removed = [x for x in args_removed if invalid not in x.lower()]
            args_removed = [x for x in args_removed if x is not '']
            clean_opcodes.append(",".join(args_removed))
        del df['bytecode']
        df = df.assign(opcode=clean_opcodes)
        print("\t-------------------------------------------------")
        print("#\tExtracting function names.")
        function_name_column = []
        for source_code in df['solidity']:
            iterator = source_code.split("function ")
            function_name = [elem.split("(")[0] for elem in iterator if ' ' not in elem.split("(")[0]]
            function_name = list(dict.fromkeys(function_name))
            function_name = ",".join([name for name in function_name if name is not '']).lower()
            function_name_column.append(function_name)
        df = df.assign(function_names=function_name_column)
        print("\t-------------------------------------------------")
        print("#\tWriting the result into a file.")
        file_name = os.path.join(self.processed_path, fn)
        df.to_csv(file_name, sep='\t', index=False)
        print("-----------------------------------------------------")
        print()

    def get_resource_code(self, sc_addr):
        api = Contract(address=str(sc_addr), api_key=self.key)
        sourcecode = api.get_sourcecode()
        return sourcecode[0]['SourceCode']

    def check_exceptions(self, fn):
        print("- Processing file: {}".format(fn))
        fp = os.path.join(self.processed_path, fn)
        df = pd.read_csv(fp, sep='\t')
        prev_len = len(df)
        to_check = df[df['solidity'] == 'Exception']
        results = dict(changed=0, same=0)
        for index, row in to_check.iterrows():
            try:
                solidity = self.get_resource_code(row['address'])
                if solidity == '':
                    solidity = 'None'
                results['changed'] += 1
            except Exception as e:
                print("\t - Cannot obtain solidity for addr: {}. Exception: {}".format(row['address'], e))
                solidity = 'Exception'
                results['same'] += 1
            df.loc[df['address'] == row['address'], 'solidity'] = solidity
        after_len = len(df)
        assert(prev_len == after_len)
        nfp = os.path.join(self.exceptions_fixed_path, fn)
        df.to_csv(nfp, sep='\t', index=False)
        print("\t - {} results changed, {} stayed the same.".format(results['changed'], results['same']))

    def preprocess_data(self):
        raw_file_names = set(os.listdir(self.raw_path))
        processed_file_names = set(os.listdir(self.processed_path))
        file_names = list(raw_file_names - processed_file_names)
        file_names = natsort.natsorted(file_names, reverse=False)
        threads = list()
        for fn in file_names:
            self.process_file(fn)
        #     t = Thread(target=self.process_file, args=(fn,))
        #     threads.append(t)
        #
        # for t in threads:
        #     t.start()

    def fix_exception_data(self):
        processed_file_names = os.listdir(self.processed_path)
        processed_file_names = [fn for fn in processed_file_names if 'test' in fn]
        file_names = natsort.natsorted(processed_file_names, reverse=False)
        for fn in file_names:
            self.check_exceptions(fn)

    def distribute_data(self):
        processed_file_names = os.listdir(self.exceptions_fixed_path)
        processed_file_names = [fn for fn in processed_file_names if 'test' in fn]
        file_names = natsort.natsorted(processed_file_names, reverse=False)
        total_open = 0
        total_close = 0
        for fn in file_names:
            print("- Processing file: {}".format(fn))
            fp = os.path.join(self.exceptions_fixed_path, fn)
            df = pd.read_csv(fp, sep='\t')
            application_set = df[df['solidity'] == 'Exception']
            application_set = application_set.append(df[df['solidity'] == 'None'])
            training_set = df[(df['solidity'] != 'None') & (df['solidity'] != 'Exception')]
            print("\tClose contracts: {}, open contracts: {}".format(len(application_set), len(training_set)))
            assert(len(df) == len(application_set) + len(training_set))
            training_path = os.path.join(self.dataset_path, "training_data")
            with open(training_path, 'a') as f:
                training_set.to_csv(f, index=False, sep='\t', header=f.tell() == 0)

            application_path = os.path.join(self.dataset_path, "application_data")
            with open(application_path, 'a') as f:
                application_set.to_csv(f, index=False, sep='\t', header=f.tell() == 0)
            total_open += len(training_set)
            total_close += len(application_set)
            print("-----------------------------------------------------")
        print()
        print("In total of {} close contracts, {} open contracts found in total dataset.".format(total_close, total_open))


if __name__ == "__main__":
    rcc = Preprocessor()
    rcc.preprocess_data()
    # note: only when all the data is preprocessed, run below.
    rcc.fix_exception_data()
    rcc.distribute_data()
