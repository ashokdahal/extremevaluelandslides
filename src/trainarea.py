from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os,datetime
import wandb
from wandb.keras import WandbCallback
import tensorflow_probability as tfp
tfd = tfp.distributions
alpha = 0.8
beta=0.2
epsilon=1e-7
gamma = 0.75

gen_par=tfp.distributions.GeneralizedPareto(0.0, 1.000343, 3.21815, validate_args=False, allow_nan_stats=False, name=None)

wandb.init(project="EQGNN")
FTP=tf.keras.metrics.TruePositives()
FFP=tf.keras.metrics.FalsePositives()
FFN=tf.keras.metrics.FalseNegatives()
KLD=tf.keras.losses.KLDivergence()
MSLE=tf.keras.losses.MeanSquaredLogarithmicError()
POI=tf.keras.losses.Poisson()

def arealoss(ytrue,ypred):
    # ytrue_prob=gen_par.prob(ytrue)
    # ypred_prob=gen_par.prob(ypred)
    n=tf.math.count_nonzero(ytrue,dtype=tf.dtypes.float32)

    return tf.math.divide_no_nan(tf.reduce_sum(tf.math.abs(tf.math.subtract(ytrue,ypred))),n)

def countloss(ytrue,ypred):
    ypred_round=tf.round(ypred)
    POI_loss=POI(ytrue,ypred_round)
    return tf.reduce_sum(POI_loss)

def tversky(y_true, y_pred):
    TP=FTP(y_true,y_pred)
    FP=FFP(y_true,y_pred)
    FN=FFN(y_true,y_pred)
    return tf.math.divide_no_nan(TP,tf.add_n([TP,tf.multiply(alpha,FN),tf.multiply(beta,FP)]))
def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    return tf.reduce_sum(tf.pow(tf.subtract(1.0,pt_1), gamma))

def compileModel(model,args):
    init_lr=args['initial_lr']
    decay_step=args['decay_step']
    decay_rate=args['decay_rate']
    focal_gamma=args['focal_gamma']
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=init_lr,decay_steps=decay_step,decay_rate=decay_rate)
    opt = Adam(learning_rate=init_lr)
    model.compile(optimizer=opt,
    loss={'area':arealoss}, 
    loss_weights={'area':10.0},
    metrics={'area':'mae'})
    return model
def trainmodel(model,xdata,ydata,args):
    tf.keras.utils.plot_model(model, "plot_model.png")
    NUMBER_EPOCHS = args['Nepoch']
    filepath=args['ckpt']
    BATCH_SIZE=args['batch']
    
    model_checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        save_freq="epoch",
        options=None
    )
    early_stop=tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=50,
        verbose=0,
        mode="max",
        baseline=None,
        restore_best_weights=True,
    )
    logdir = os.path.join(args["tensorboard_log"]+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    wandb_callback = WandbCallback(monitor='val_loss',
                               log_weights=True,
                               log_evaluation=True,
                               validation_steps=5)
    hist = model.fit(x=xdata,
                     y=ydata,
                     epochs=NUMBER_EPOCHS,
                     batch_size=BATCH_SIZE,
                     validation_split=0.2,#auto validate using 30% of random samples at each epoch
                     verbose=1, callbacks=[model_checkpoint_callback,tensorboard_callback,early_stop,wandb_callback]
                    )
    return hist
