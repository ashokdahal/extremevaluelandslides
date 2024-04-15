from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from scipy.stats import genpareto
# import tensorflow_probability as tfp
import numpy as np
from scipy.stats import genpareto

# import tensorflow_probability.distributions as tfp
# tfd = tfp.distributions
from keras import backend as K


class lhmodel:
    def __init__(self, modelparam):
        self.depth = modelparam["depth"]
        self.infeatures = modelparam["infeatures"]
        self.outfeatures = modelparam["outfeatures"]
        self.units = modelparam["units"]
        self.kernel_initializer = modelparam["kernel_initializer"]
        self.bias_initializer = modelparam["bias_initializer"]
        self.droupout = modelparam["droupout"]
        self.batchnormalization = modelparam["batchnormalization"]
        self.dropoutratio = modelparam["dropoutratio"]
        self.lastactivation = modelparam["lastactivation"]
        self.middleactivation = modelparam["middleactivation"]
        self.lr = modelparam["lr"]
        self.decay_steps = modelparam["decay_steps"]
        self.decay_rate = modelparam["decay_rate"]
        self.landslideweight = modelparam["weight_landslide"]
        self.nolandslideweight = modelparam["weight_nolandslide"]
        self.opt = tf.keras.optimizers.Adam
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.poisson = tf.keras.losses.Poisson(name="poisson")
        self.su_weight = modelparam["su_weight"]
        self.ad_weight = modelparam["ad_weight"]
        self.fr_weight = modelparam["fr_weight"]
        self.auc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)

    def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def getAreaDensityModel(self):
        features_only = Input((self.infeatures))
        x = layers.Dense(
            units=self.units,
            activation="relu",
            name=f"AR_DN_0",
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )(features_only)
        for i in range(1, self.depth + 1):
            x = layers.Dense(
                activation="relu",
                units=self.units,
                name=f"AR_DN_{str(i)}",
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
            )(x)
            if self.batchnormalization:
                x = layers.BatchNormalization()(x)
            if self.droupout:
                x = layers.Dropout(self.dropoutratio)(x)

        out_areaDen = layers.Dense(
            units=self.outfeatures, activation="relu", name="areaout"
        )(x)

        out_sus = layers.Dense(
            units=self.outfeatures, activation="sigmoid", name="susout"
        )(x)
        self.model = Model(inputs=features_only, outputs=[out_sus,out_areaDen])

    def getOptimizer(
        self,
    ):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    def dice_loss(self, y_true, y_pred, smooth=1e-6):  # Only Smooth
        """
        https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
        """
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        smooth = tf.cast(smooth, y_pred.dtype)

        intersection = tf.math.reduce_sum(tf.math.abs(y_true * y_pred), axis=-1)
        dice_coef = (2.0 * intersection + smooth) / (
            tf.math.reduce_sum(tf.math.square(y_true), -1)
            + tf.math.reduce_sum(tf.math.square(y_pred), -1)
            + smooth
        )
        return 1 - dice_coef
    def egpdloss(self, ytrue, ypred):
        #'landslide','area_density','count'
        # ______Probability Part_________
        ypred=ypred[ytrue>0.1]
        ytrue=ytrue[ytrue>0.1]
        kappa = 0.8118067
        sig = tf.nn.relu(ypred)+0.2
        xi = 0.4919825
        
        y = tf.nn.relu(ytrue)
        
        sig = sig - sig * (1 - tf.math.sign(y)) + (1 - tf.math.sign(y))  # If no exceedance, set sig to 1
        kappa = kappa - kappa * (1 - tf.math.sign(y)) + (1 - tf.math.sign(y))  # If no exceedance, set kappa to 1
        xi = xi - xi * (1 - tf.math.sign(y)) + (1 - tf.math.sign(y))  # If no exceedance, set xi to 1
        
        # Evaluate log-likelihood
        ll1 = -(1 / xi + 1) * tf.math.log(1 + xi * y / sig)
        
        # Uses non-zero response values only
        ll2 = tf.math.log(sig) * tf.math.sign(ll1)
        
        ll3 = -tf.math.log(kappa) * tf.math.sign(ll1)
        
        y = y - y * (1 - tf.math.sign(y)) + (1 - tf.math.sign(y))  # If zero, set y to 1
        
        ll4 = (kappa - 1) * tf.math.log(1 - (1 + xi * y / sig)**(-1 / xi))
    
        return -tf.reduce_sum(ll1 + ll2 + ll3 + ll4)
        
    def f1_m(self, ytrue, ypred):
        y_true=ytrue
        y_pred=ypred
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def preparemodel(self, weights=None):
        self.getAreaDensityModel()
        self.getOptimizer()
        self.model.compile(
            optimizer=self.optimizer,
            loss=['binary_crossentropy',self.egpdloss],
            loss_weights=[100,1],
            metrics=[self.f1_m,tf.keras.metrics.RootMeanSquaredError()],
        )