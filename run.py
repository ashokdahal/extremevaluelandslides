import tensorflow as tf
import json
from src import preparedata
from src import modelarea
from src import trainarea

print(tf.__version__)
print(tf.config.list_physical_devices(
    device_type=None
))

params=json.load(open('params/params.json','r'))
dataset=preparedata.readGPDData(params['dataprepinargs'])
dataset.preparedata()

landslidehazard=modelarea.lhmodel(params['modelparam'])
landslidehazard.preparemodel()

trainarea.trainmodel(landslidehazard.model,dataset.X_train,dataset.Y_train,params['trainparam'])
