import os
import sys
import time
b1 = int(sys.argv[1])
b2 = int(sys.argv[2])

option = sys.argv[3]

for i in range(b1, b2):
    if option=='r':
        os.system('python3 train_elecsim_process.py {} r 0'.format(i))
        os.system('python3 train_elecsim_process.py {} r 1'.format(i))
    elif option=='i':
        time.sleep(40)
        os.system('python3 train_elecsim_process.py {} i'.format(i))

