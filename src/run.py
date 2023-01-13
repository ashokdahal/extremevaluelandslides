
import numpy as np
import seaborn as sns
from EQALS import SimIO
import json
import os
from EQALS import getGraphs
from EQALS import trainall as train
from EQALS import ClassificationModel
from tensorflow.keras import Sequential, Model

params=json.load(open('params.json','r'))


dataobj=SimIO.readSalvus(params['salvus_inargs'])
dataobj.read()
#dataobj.addPeak()
graphs=getGraphs.prepareGraphDataSalvus(dataobj,params['grargs'])
# graphs=getGraphs.prepareGraphData(dataobj,params['grargs'])
print('---------------Data Reading Complete---------------')
graphs.PrepareData()
print('---------------Data Preparation Complete---------------')
clsargs=params['clsargs']
gcn_GRU=ClassificationModel.GCN_GRU(
    seq_len=clsargs['seq_len'],
    adj=np.ones((16,16)),
    gc_layer_sizes=clsargs["gc_layer_sizes"],
    gc_activations=clsargs["gc_activations"],
    GRU_layer_sizes=clsargs["GRU_layer_sizes"],
    GRU_activations=clsargs["GRU_activations"],
)
x_input, x_output = gcn_GRU.in_out_tensors()
model = Model(inputs=x_input, outputs=x_output)
print(model.summary())
print('---------------model prepared---------------')
x_input, x_output = gcn_GRU.in_out_tensors()
model = Model(inputs=x_input, outputs=x_output)
model=train.compileModel(model,params["trainargs"])


print('---------------model compiled ---------------')
print('---------------Training Started ---------------')
sus_ytrain=np.expand_dims(graphs.YTrain_mat[:,:,0],axis=-1)
area_ytrain=np.expand_dims(graphs.YTrain_mat[:,:,1],axis=-1)
count_ytrain=np.expand_dims(graphs.YTrain_mat[:,:,2],axis=-1)
print(f'training wiht shapes of sus, area, count {sus_ytrain.shape,area_ytrain.shape,count_ytrain.shape}')
print(f'training wiht max of sus, area, count {sus_ytrain.max(),area_ytrain.max(),count_ytrain.max()}')
hist=train.trainmodel(model,[graphs.XTrain_mat,graphs.adj_mat,graphs.XTrain_mat_cov],[sus_ytrain,area_ytrain,count_ytrain],params["trainargs"])

print('---------------Training Completed ---------------')
text_file = open("History.txt", "w")
 
#write string to file
text_file.write(hist)
 
#close file
text_file.close()
print('---------------History Saved ---------------')
print('---------------EXITING ---------------')