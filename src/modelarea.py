from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from scipy.stats import genpareto
import tensorflow_probability as tfp
import numpy as np
from scipy.stats import genpareto
#import tensorflow_probability.distributions as tfp
tfd = tfp.distributions

class lhmodel():
    def __init__(self,modelparam):
        self.depth=modelparam['depth']
        self.infeatures=modelparam['infeatures']
        self.outfeatures=modelparam['outfeatures']
        self.units=modelparam['units']
        self.kernel_initializer=modelparam['kernel_initializer']
        self.bias_initializer=modelparam['bias_initializer']
        self.droupout=modelparam['droupout']
        self.batchnormalization=modelparam['batchnormalization']
        self.dropoutratio=modelparam['dropoutratio']
        self.lastactivation=modelparam['lastactivation']
        self.middleactivation=modelparam['middleactivation']
        self.lr=modelparam['lr']
        self.decay_steps=modelparam['decay_steps']
        self.decay_rate=modelparam['decay_rate']
        self.landslideweight=modelparam['weight_landslide']
        self.nolandslideweight=modelparam['weight_nolandslide']
        self.opt=tf.keras.optimizers.Adam

    def getAreaDensityModel(self):

        features_only=Input((self.infeatures))
        x=layers.Dense(units=self.units,activation='selu',name=f'AR_DN_0',kernel_initializer=self.kernel_initializer,bias_initializer=self.bias_initializer)(features_only)
        for i in range(1,self.depth+1):
            x=layers.Dense(activation=None,units=self.units,name=f'AR_DN_{str(i)}',kernel_initializer=self.kernel_initializer,bias_initializer=self.bias_initializer)(x)
            if self.batchnormalization:
                x= layers.BatchNormalization()(x)
            if self.droupout:
                x= layers.Dropout(self.dropoutratio)(x) 
            x=layers.LeakyReLU(alpha=0.2)(x)     
        out_areaDen=layers.Dense(units=self.outfeatures,activation='exponential',name='areaDen')(x)
        self.model = Model(inputs=features_only, outputs=out_areaDen)

    def getOptimizer(self,):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.lr,decay_steps=self.decay_steps,decay_rate=self.decay_rate,staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # def gevloss(self,ytrue,ypred):
    #     loc=ypred[:,0]
    #     scale=ypred[:,1]
    #     conc=ypred[:,1]
    #     dist=tfp.distributions.GeneralizedExtremeValue(loc=loc,scale=0.1,concentration=conc,validate_args=False,allow_nan_stats=False,name='GeneralizedExtremeValue')
    #     lik=-dist.log_prob(ytrue)
    #     return tf.reduce_mean(lik)
    # def gevmetric(self,ytrue,ypred):
    #     loc=ypred[:,0]
    #     scale=ypred[:,1]
    #     conc=ypred[:,1]
    #     dist=tfp.distributions.GeneralizedExtremeValue(loc=loc,scale=0.1,concentration=conc,validate_args=False,allow_nan_stats=False,name='GeneralizedExtremeValue')
    #     lik=dist.prob(ytrue)
    #     return tf.reduce_mean(lik)
    def gpdloss(self,ytrue,ypred):
        loc=0.0
        scale=ypred[:,0]
        conc=ypred[:,1]
        weight=tf.cast(ytrue>0,dtype=tf.dtypes.float32)
        weight=(weight*(self.landslideweight-self.nolandslideweight))+self.landslideweight
        dist=tfp.distributions.GeneralizedPareto(loc=loc,scale=scale,concentration=conc,validate_args=False,allow_nan_stats=False,name='GeneralizedExtremeValue')
        lik=-dist.log_prob(ytrue)
        return tf.reduce_mean(tf.math.multiply(lik,weight))
    def gpdmetric(self,ytrue,ypred):
        loc=0.0
        scale=ypred[:,0]
        conc=ypred[:,1]
        dist=tfp.distributions.GeneralizedPareto(loc=loc,scale=scale,concentration=conc,validate_args=False,allow_nan_stats=False,name='GeneralizedExtremeValue')
        lik=dist.prob(ytrue)
        return tf.reduce_mean(lik)

    def preparemodel(self,weights=None):
        self.getAreaDensityModel()
        self.getOptimizer()
        self.model.compile(optimizer=self.optimizer, loss=self.gpdloss, metrics=self.gpdmetric)
    
'''
#old version of code only used for reference. 


class ADModel():
    def __init__(self,loc,scale,shape):
        self.sigma=scale
        self.shi=shape
        self.loc=tf.constant(loc)
        self.scale=tf.constant(scale)
        self.concentration=tf.constant(shape)
        self.depth=64
        self.alpha=0.0
        self.beta=1.0
        self.N_STEPS=200
    def getAreaDensityModel2(self,in_num=17,out_num=1):
        features_only=Input((17,1))
        #x=layers.GRU(units=256,name='GRU')(features_only)
        x=layers.Conv1D(filters=32,kernel_size=3,strides=1,padding="valid",data_format="channels_last",activation='relu',name='AR_CN_first',kernel_initializer='random_normal',bias_initializer='random_uniform')(features_only)
        x= layers.BatchNormalization()(x)
        x=layers.Activation('selu')(x)
        for i in range(32):
            x=layers.Conv1D(filters=32,kernel_size=3,strides=1,padding="valid",data_format="channels_last",name=f'AR_CN_{str(i+1)}',kernel_initializer='random_normal',bias_initializer='random_uniform')(features_only)
            x= layers.BatchNormalization()(x)
            x=layers.Activation('selu')(x)
        x=layers.Flatten()(x)

        x=layers.Dense(units=1,name=f'AR_DN_2',activation='tanh')(x)
        self.model = Model(inputs=features_only, outputs=x)

    def getAreaDensityModel(self,in_num=17,out_num=1):

        features_only=Input((in_num))

        x=layers.Dense(units=64,name=f'AR_DN_0',kernel_initializer='he_normal',bias_initializer='he_uniform')(features_only)
        for i in range(1,self.depth+1):
            x=layers.Dense(units=64,name=f'AR_DN_{str(i)}',kernel_initializer='he_normal',bias_initializer='he_uniform')(x)
            x= layers.BatchNormalization()(x)
            x= layers.Dropout(.3)(x)
            x=layers.Activation('selu')(x)
            

        
        out_areaDen=layers.Dense(units=2,activation='sigmoid',name='areaDen')(x)
        #out_areaDenProb=GPDLayer(self.loc,self.scale,self.concentration)(x)
        #out_areaDenProb=layers.Dense(units=1,activation='relu',name='areaDenProb')(x)#tfp.layers.DistributionLambda(lambda t: tfd.GeneralizedExtremeValue(loc=t[..., 0],scale=t[...,1],concentration=t[...,2]))(x)
        self.model = Model(inputs=features_only, outputs=out_areaDen)
        #return model
    def getGPD(self):
        self.dist=tfd.GeneralizedPareto(loc=self.loc,scale=self.scale,concentration=self.concentration)

    def getOptimizer(self,opt=tf.keras.optimizers.Adam,lr=1e-3,decay_steps=10000,decay_rate=0.9):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,decay_steps=decay_steps,decay_rate=decay_rate)
        self.optimizer = opt(learning_rate=1e-3)
    
    def gpdLoss(self,ytrue,ypred):
        first_term=tf.math.multiply(tf.math.subtract(ytrue,ypred),self.concentration)
        second_term=((1/self.concentration)+1)
        log_first_term=tf.math.log(tf.math.abs(tf.math.add(first_term,self.scale)))
        my_GPDLOSS=tf.math.multiply(log_first_term,second_term)
       
        return tf.math.reduce_mean(my_GPDLOSS)
    # def novelGPDLoss(self,ytrue,ypred):
    #     P=self.getGPD()
    #     p1=P.prob(ytrue).numpy()
    #     p2=P.prob(ypred).numpy()
    #     KLD=p1*np.log(p1 / p2)



    def _z(self, x, scale,loc):
        #loc = tf.convert_to_tensor(self.loc)
        return (x - loc) / scale
    
    def _log_prob(self, xt,x):
        scale = tf.convert_to_tensor(self.scale)
        loc = tf.convert_to_tensor(xt)
        concentration = tf.convert_to_tensor(self.concentration)
        z = self._z(x, scale,loc)
        eq_zero = tf.equal(concentration, 0)  # Concentration = 0 ==> Exponential.
        nonzero_conc = tf.where(eq_zero, tf.constant(1, 'float32'), concentration)
        y = 1 / nonzero_conc + tf.ones_like(z, 'float32')
        where_nonzero = tf.where(
            tf.equal(y, 0), y, y * tf.math.log1p(nonzero_conc * z))
        return -tf.math.log(scale) - tf.where(eq_zero, z, where_nonzero)


    def distributionLoss(self,ytrue,ypred):
        scale=ypred[:,0]
        conc=ypred[:,1]
        dist=tfp.distributions.GeneralizedExtremeValue(0.0,scale,coc,validate_args=False,allow_nan_stats=True,name='GeneralizedExtremeValue')
        negloglik=-dist.log_prob(ytrue)
        return tf.reduce_sum(negloglik)

    def MSE(self,ytrue,ypred):
        # dist=tfp.distributions.GeneralizedPareto(self.loc, self.scale, self.concentration, validate_args=False, allow_nan_stats=True, name=None)
        try:
            ytrue_prob=ypred.log_prob(ytrue)
        except:
            raise RuntimeError(f'eroror {ytrue}, {ypred}')
        return  ytrue_prob
    def getdist(self,ytrue,ypred):
        # cX = np.concatenate((ytrue,ypred))
        # txs = np.linspace(ytrue.min(),ytrue.max(),self.N_STEPS)
        # pxs = np.linspace(ypred.min(),ypred.max(),self.N_STEPS)
        dist=tfp.distributions.GeneralizedPareto(self.loc, self.scale, self.concentration)
        # paretopar1=genpareto.fit(ytrue)
        # paretopar2=genpareto.fit(ypred)
        p1=dist.prob(ytrue).numpy()
        p2=dist.prob(ypred).numpy()
        # dx=(cX.max()-cX.max())/self.N_STEPS
        #bht = -np.log(np.sum(np.sqrt(p1*p2)))
        #print(f'distance is {bht.max()} p1 is {p1.max()} and p2 is {p2.max()}')
        KLD=p1*np.log(p1 / p2)
        return KLD.astype(np.float32)



    def GPDBTLoss(self,ytrue,ypred):
              
        distance=tf.numpy_function(self.getdist, [ytrue,ypred], tf.float32)
        mae=tf.keras.losses.MeanAbsoluteError()(ytrue,ypred)


        return tf.math.add(tf.math.multiply(tf.reduce_sum(distance),self.alpha),tf.math.multiply(mae,self.beta))

    def compileModel(self,weights=None):
        negloglik = lambda ytr, ypr: -ypr
        losses = {'areaDenProb': negloglik, 'areaDen': 'mae'}
        mertrics={'areaDenProb': 'mae', 'areaDen': 'mse'}
        lossWeights = {"areaDenProb": 0.05, "areaDen": 0.95}
        self.model.compile(optimizer=self.optimizer, loss=self.distributionLoss, metrics='mae')
    
        
'''