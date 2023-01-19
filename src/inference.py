import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

def inferenceLH(model,model_weights,xdata,rp):
    model.load_weights(model_weights)
    trained_model=model
    gpd_params=trained_model.predict(xdata)
    loc=0.0
    scale=gpd_params[:,0]
    conc=gpd_params[:,1]
    dists=tfp.distributions.GeneralizedPareto(loc=0,scale=scale,concentration=conc,validate_args=False,allow_nan_stats=False,name='GeneralizedExtremeValue') 

    intensity=dists.quantile((1-1/rp)).numpy()

    frequency=1/rp

    prob_occ=1-dists.cdf(0.1).numpy()

    landslide_hazard=prob_occ*intensity*frequency

    return landslide_hazard
