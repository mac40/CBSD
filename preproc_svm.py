# USAGE
# python preproc_svm.py --data data_file.csv

import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True,
                help="path to formatted data frame/ear/eye_state")
args = vars(ap.parse_args())

dati=pd.read_csv(args["data"], sep=",", index_col="frame")

'''
dati.tag = dati.tag.where(mask, 1)
mask = dati.tag != "half"
dati.tag = dati.tag.where(mask, 0)
'''
listear=list(dati.ear)
col=['F1',"F2","F3","F4","F5",'F6',"F7","F8","F9","F10",'F11',"F12","F13"]
df_fin=pd.DataFrame(columns=col)

for i in range(6, len(listear)-7):
    tmp_ear=listear[i-6:i+7]

    '''
    tmp_dict=dict()
    for j in range(0,6):
        tmp_dict[col[j]]=tmp_ear[j]
    '''
    tmp_ear.append(tmp_tag)
    #df_tmp=pd.DataFrame(data=[tmp_ear], columns=col)
    df_fin.loc[i]=tmp_ear

df_fin.index.name="frame"

df_fin = df_fin[df_fin.F1!=0]
df_fin = df_fin[df_fin.F2!=0]
df_fin = df_fin[df_fin.F3!=0]
df_fin = df_fin[df_fin.F4!=0]
df_fin = df_fin[df_fin.F5!=0]
df_fin = df_fin[df_fin.F6!=0]
df_fin = df_fin[df_fin.F7!=0]
df_fin = df_fin[df_fin.F8!=0]
df_fin = df_fin[df_fin.F9!=0]
df_fin = df_fin[df_fin.F10!=0]
df_fin = df_fin[df_fin.F11!=0]
df_fin = df_fin[df_fin.F12!=0]
df_fin = df_fin[df_fin.F13!=0]

df_fin.to_csv("preprocessed_noNorm/preproc_{}".format(args["data"][11:]), index=True, header=True)
