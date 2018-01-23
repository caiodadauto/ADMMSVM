import time
import subprocess as sub
from traindatas import datas_conf

def start():
    delay = 3
    sub.check_call('clear', shell = True)

    while(True):
        print("Available set of data for the test")
        for data_set in datas_conf.keys():
            print("\t> " + data_set)
        data_key = input("Choose what data will be used for the test: ")
        if data_key in datas_conf:
            break
        else:
            sub.check_call('clear', shell = True)
            print("Insert one of the available datas!")
            time.sleep(delay)
            sub.check_call('clear', shell = True)

    sub.check_call('clear', shell = True)

    while(True):
        linear_or_not = input("What is the type of the classifier (linear or nonlinear): ")
        if linear_or_not == "linear" or linear_or_not == "nonlinear":
            sub.check_call('clear', shell = True)
            break
        else:
            sub.check_call('clear', shell = True)
            print("Choose between linear or nonlienar!")
            time.sleep(delay)
            sub.check_call('clear', shell = True)
            continue

    if linear_or_not == "linear":
        while(True):
            try:
                nodes    = int(input("How many nodes are there in your network? "))
            except ValueError:
                sub.check_call('clear', shell = True)
                print("The number of network must be a positive whole number!")
                time.sleep(delay)
                sub.check_call('clear', shell = True)
                continue

            if nodes >= 2:
                break
            else:
                sub.check_call('clear', shell = True)
                print("The network must have at least 2 nodes!")
                time.sleep(delay)
                sub.check_call('clear', shell = True)
    else:
        print("For nonlinear classifier will be used 18 nodes!")
        nodes = 18
        time.sleep(delay)

    sub.check_call('clear', shell = True)

    return [nodes, linear_or_not, datas_conf[data_key]]
