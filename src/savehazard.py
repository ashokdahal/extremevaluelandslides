import geopandas as gpd
import pandas as pd 
import numpy as np

def save_predicted_Haz(ids,prediction,hazcol,sufile,outfile):
    su=gpd.read_file(sufile)
    df=pd.DataFrame({'cat':ids,hazcol:prediction})
    hazdata=su.merge(df,on='cat')
    hazdata.to_file(outfile)