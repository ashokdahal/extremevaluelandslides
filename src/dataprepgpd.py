import numpy as np
import pandas as pd
from tqdm import tqdm
class readGPDData():
    def __init__(self,features_folder,target_folder):
        self.features_folder=features_folder
        self.target_folder=target_folder
        self.readFeatures()
        self.readTargets()
    def readFeatures(self):
        for i in tqdm(range(31)):
            if (1988+i==2011) | (1988+i==2012):
                continue
            features_data=pd.read_csv(f'{self.features_folder}/covars_{str(1988+i)}.csv',index_col=0).set_index('Cat')
            features_data=features_data.fillna(0)
            features_data=(features_data-features_data.mean())/features_data.std()
            if i==0:
                Xtrain=features_data.to_numpy()
            else:
                Xtrain=np.vstack((Xtrain,features_data.to_numpy()))
        self.Xtrain=Xtrain

    def readTargets(self):
        for i in tqdm(range(31)):
            if (1988+i==2011) | (1988+i==2012):
                continue
            target_data=pd.read_csv(f'{self.target_folder}/targets_{str(1988+i)}.csv',index_col=0).set_index('cat')
            if i ==0:
                #count=target_data['number_landslide'].to_numpy()
                aread=target_data['area_density'].to_numpy()
            else:
                #count=np.hstack((count,target_data['number_landslide'].to_numpy()))
                aread=np.hstack((aread,target_data['area_density'].to_numpy()))
        self.Ytrain=aread

    def removeZeros(self):
        idx=self.Ytrain!=0
        self.Ytrain=self.Ytrain[idx]
        self.Xtrain=self.Xtrain[idx]