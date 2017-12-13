# USAGE
# python preproc_svm_normalizzato_new_video.py --data non_training_data_raw_data/data_file.csv

import argparse
import pandas as pd
import numpy as np

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

#normalizzo
listear=np.array(listear)
listear=(listear-np.nanmin(listear))/(np.nanmax(listear)-np.nanmin(listear))
listear=list(listear)

col=['F1',"F2","F3","F4","F5",'F6',"F7","F8","F9","F10",'F11',"F12","F13"]
df_fin=pd.DataFrame(columns=col)


for i in range(6, len(listear)-7):
    tmp_ear=listear[i-6:i+7]
    df_fin.loc[i]=tmp_ear
	
df_fin.index.name="frame"
df_fin.dropna(how='any', inplace=True)
df_fin.to_csv("non_training_data_preproc/preproc_{}".format(args["data"][27:]), index=True, header=True)
