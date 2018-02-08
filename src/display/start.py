import time
import subprocess as sub
from tests import tests_conf

def start():
    delay = 3
    sub.check_call('clear', shell = True)

    while(True):
        print("Available set of tests")
        i    = 1
        keys = list(tests_conf.keys())
        for key in keys:
            print("\t" + str(i) + ") " + key)
            i += 1
        n = int(input("Choose what the number of the test will be done: "))
        try:
            key = keys[n - 1]
            break
        except IndexError:
            sub.check_call('clear', shell = True)
            print("Insert one of the available numbers!")
            time.sleep(delay)
            sub.check_call('clear', shell = True)

    sub.check_call('clear', shell = True)

    if tests_conf[key]['get_nodes']:
        while(True):
            nodes    = int(input("How many nodes are there in your network? "))
            if nodes < 0:
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
        if tests_conf[key]['data_info']['file'] == 'circles':
            print("For circles test will be used 4 nodes!")
            nodes = 4
        elif tests_conf[key]['data_info']['file'] == 'chess':
            print("For chess test will be used 8 nodes!")
            nodes = 8
        else:
            print("For pima indians diabetes test will be used 6 nodes!")
            nodes = 6
        time.sleep(delay)

    sub.check_call('clear', shell = True)

    return [nodes, tests_conf[key]]
