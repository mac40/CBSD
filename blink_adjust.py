'''
change 13 frames of a blink in 1 frame

USAGE: python blink_adjust.py -d file_name.py
'''
import argparse
import pandas as pd

AP = argparse.ArgumentParser()
AP.add_argument("-d", "--data", required=True,
                help="data to be changed")
ARGS = vars(AP.parse_args())

DATA = pd.read_csv(ARGS["data"], sep=",", index_col=0)
FRAME_LIST = list(DATA.index)
BLINK_LIST = list(DATA.blink)

TAG = 0

for BLINK in range(0, len(BLINK_LIST)):
    if BLINK_LIST[BLINK] == 1.0:
        for TEMP in range(BLINK+1, BLINK+13):
            if BLINK_LIST[TEMP] == 0.0:
                TAG += 1
        if TAG > 4:
            for TEMP in range(BLINK, BLINK + 13):
                BLINK_LIST[TEMP] = 0.0
            TAG = 0
        else:
            for TEMP in range(BLINK + 1, BLINK + 13):
                BLINK_LIST[TEMP] = 0.0
            TAG = 0
BLINK_LIST = pd.DataFrame(BLINK_LIST, index=FRAME_LIST)
BLINK_LIST.index.name='frame'
BLINK_LIST.columns = ['blink']
BLINK_LIST.to_csv("results.csv")
