'''
change 13 frames of a blink in 1 frame

USAGE: python blink_adjust.py -d file_name.py
'''
import argparse
import pandas as pd
import copy

AP = argparse.ArgumentParser()
AP.add_argument("-d", "--data", required=True,
                help="data to be changed")
ARGS = vars(AP.parse_args())

DATA = pd.read_csv(ARGS["data"], sep=",", index_col=0)
FRAME_LIST = list(DATA.index)
BLINK_LIST = list(DATA.blink)

for n in range(len(BLINK_LIST)):
    #trovo il primo 1.0
    if BLINK_LIST[n]==1.0:
        i = copy.deepcopy(n)
        #correggi 1.0 isolati: se è un 1.0 singolo o doppio diventa 0.0
        if sum(BLINK_LIST[i:i+5])<=2.0:
            BLINK_LIST[i]=0.0
        else:
            #correggi 0.0 isolati: se ci sono 0.0 singoli o doppi appena dopo diventano 1.0
            while (sum(BLINK_LIST[i:i+5])>2.0):
                BLINK_LIST[i]=1.0
                BLINK_LIST[i+1]=1.0
                BLINK_LIST[i+2]=1.0
                i+=1

#ora ho solo 1.0 consecutivi parto a cercare dal fondo così elimino man mano e poi aggiungo 6 frame
for n in range(len(BLINK_LIST)):
    #trovo il primo 1.0
    if BLINK_LIST[n]==1.0:
        i = copy.deepcopy(n)
        while (BLINK_LIST[i+1]==1.0):
            BLINK_LIST[i+1]=0.0
            i+=1

#scala gli 1.0 di 8 frame
BLINK_LIST=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]+BLINK_LIST[:len(BLINK_LIST)-8]


BLINK_LIST = pd.DataFrame(BLINK_LIST, index=FRAME_LIST)
BLINK_LIST.index.name='frame'
BLINK_LIST.columns = ['blink']
BLINK_LIST.to_csv("results_prova.csv")
