import pandas as pd
import numpy as np
import scipy.stats as stats


url_data="xxx.csv"
url_base="xxx.csv"
dataset = pd.read_csv(url_data, index_col="frame")
baseline = pd.read_csv(url_base, index_col="frame")
#array di dati
data=dataset.blink.values
base=baseline.blink.values
rate_medio=sum(base)/len(base)
rate_IC = rate_medio + np.array([-1,1])*np.std(base)*stats.norm.ppf(0.975)/np.sqrt(len(base))
delta_frame=int(1/rate_medio)

for i in range(len(data)):
    if data[i]==1.0:
        data[i]+=-1
        #finestra separazione blink sinistra e lunga 2delta
        if (i-delta_frame+1)<0:
                a=0
        else:
            a=i-delta_frame+1
        for j in range(a,i+1):
            data[j]+=1.0/(delta_frame)


y=np.zeros(delta_frame)
for k in range (delta_frame,len(data)):
    y=np.append(y,sum(data[k-delta_frame:k])-1)

newdata = pd.DataFrame(y, index=dataset.index, columns=["y"])
newdata.to_csv("rate_prova.csv")

