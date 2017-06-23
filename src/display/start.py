import subprocess as sub
from datasfortest.datasconf import datas_conf

def start():
    sub.check_call('clear', shell = True)

    print("Available set of data fpr test")
    for data_set in datas_conf.keys():
        print("\t> " + data_set)

    data_key = input("Choose what data will be used for the test: ")
    nodes    = int(input("\nHow many nodes are there in your network: "))

    return [nodes, datas_conf[data_key]]
