import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics
from data import load_data
from models import *
import os


#load everything required
x = load_data("data/x_array_qqq.npy", noise_dims = 0)
x_data_qq = np.load("data/x_parametrized_data_qq.npy")
y_data_qq = np.load("data/y_parametrized_data_qq.npy")
model_qq = tf.keras.models.load_model("model_qq_opt2")

#Loop over signal injection amounts M
#For a given signal injection amount, inject events according to N ~ Poission(M)
#For a given N injected events, initialize the network with w ~ Uniform.  Do this k times.

qq = "qq"

epsilon = 1e-4
k_runs = 5
noise = False
ran_once = False

feature_dims = 6
params = 3

m1 = 2
m2 = 5
test_signal = int(1/2*len(x[m1,m2, qq, noise]))

model_full = train_supervised(feature_dims, m1, m2)

msic1_runs = []
msic2_runs = []
msic1_stds = []
msic2_stds = []

initial_weights_runs = []

sigspace = np.logspace(-3,-1,10)
for sigfrac in sigspace:
    
    initial_weights = []
    
    msic1_median = []
    msic2_median = []
    
    print(f"---------------{sigfrac}---------------")
    
    # N ~ Poission(M) injected events
    for injection in range(len(sigspace)):
        print(f"Injecting N = {len(sigspace) - injection} times, currently on: N = {injection}")
        
        #randomized signal
        random_test_signal_length = random.randint(0, test_signal - 1)
        N = int(1/4 * (len(x[0,0, qq, noise])))
        signal = x[m1, m2, qq, noise][random_test_signal_length:random_test_signal_length + int(sigfrac*N)]
        
        #fixed signal portion
        # N = int(1/4 * (len(x[0,0, qq, noise])))
        # signal = x[m1, m2, qq, noise][test_signal:test_signal + int(sigfrac*N)]
        
        msic1_kruns = []
        msic2_kruns = []
        
        for k in range(k_runs):
            print(f"Ensembling {k_runs} for Signal Fraction : {sigfrac}")

            w1 = round(random.uniform(0.5, 6),3)
            w2 = round(random.uniform(0.5, 6),3)
            initial_weights.append((w1, w2))
            print(f"Initialization: {w1} {w2}")

            model_semiweak = compileSemiWeakly(model_qq, feature_dims, params, m1, m2, w1, w2)

            test_background = int(1/2 * len(x[0,0, qq, noise]))
            train_reference = int(1/4 *len(x[0,0, qq, noise]))
            train_data = int(1/4 * len(x[0,0, qq, noise]))
            test_signal = int(1/2*len(x[m1,m2, qq, noise]))

            x_data_ = np.concatenate([x[0,0, qq, noise][test_background:],signal])
            y_data_ = np.concatenate([np.zeros(train_reference),np.ones(train_data),np.ones(len(signal))])

            X_train_, X_val_, Y_train_, Y_val_ = train_test_split(x_data_, y_data_, test_size=0.5, random_state = 42)

            history_semiweak = model_semiweak.fit(X_train_[:,0:feature_dims], Y_train_, epochs=100,
                                                   validation_data=(X_val_[:,0:feature_dims], Y_val_),batch_size=1024, verbose = 0)

            print(f"m1: {m1}",f"m2: {m2}", f"w1: {model_semiweak.trainable_weights[0].numpy()[0][0]}", f"w2: {model_semiweak.trainable_weights[1].numpy()[0][0]}")

            scores = model_semiweak.predict(np.concatenate([x[0,0, qq, noise][0:test_background],x[m1,m2, qq, noise][0:test_signal]]),batch_size=1024)
            y = np.concatenate([np.zeros(test_background),np.ones(test_signal)])
            fpr, tpr, _ = metrics.roc_curve(y, scores)

            msic1_value = np.max(tpr/np.sqrt(fpr+epsilon))
            print(f"Max SIC k = {k}: ", msic1_value)

            #weakly supervised
            model_cwola = compileCWOLA(feature_dims, m1, m2)
            myhistory_cwola = model_cwola.fit(X_train_[:,0:feature_dims], Y_train_, epochs=10,validation_data=(X_val_[:,0:feature_dims], Y_val_),batch_size=1024, verbose = 0)

            scores2 = model_cwola.predict(np.concatenate([x[0,0, qq, noise][0:test_background],x[m1,m2, qq, noise][0:test_signal]]),batch_size=1024)
            y2 = np.concatenate([np.zeros(test_background),np.ones(test_signal)])
            fpr2, tpr2, _ = metrics.roc_curve(y2, scores2)

            #fully supervised dedicated training initiaized once (this still predicts for every s/b can be simplified)
            if ran_once == False:
                scores_full = model_full.predict(np.concatenate([x[0,0, qq, noise][0:test_background],x[m1,m2, qq, noise][0:test_signal]]),batch_size=1024)
                y_full = np.concatenate([np.zeros(test_background),np.ones(test_signal)])
                fpr_full, tpr_full, _ = metrics.roc_curve(y_full, scores_full)

            #parametrized classifer
                scores_full2 = model_qq.predict(x_data_qq[np.product(x_data_qq[:,6:8]==[m1,m2],axis=1)==1],batch_size=1000)
                fpr_full2, tpr_full2, _ = metrics.roc_curve(y_data_qq[np.product(x_data_qq[:,6:8]==[m1,m2],axis=1)==1], scores_full2)

            ran_once = True
            #per-event probability
            msic1_kruns.append(msic1_value)
            msic2_kruns.append(np.max(tpr2/np.sqrt(fpr2+epsilon)))
        
        #average over the k classifiers runs
        msic1_median.append(np.median(msic1_kruns))
        msic2_median.append(np.median(msic2_kruns))
    
    msic1_runs.append(np.median(msic1_median))
    msic2_runs.append(np.median(msic2_median))
    msic1_stds.append(np.std(msic1_median))
    msic2_stds.append(np.std(msic2_median))
        
np.save("data/msic1_median_test.npy", msic1_runs)
np.save("data/msic2_median_test.npy", msic2_runs)
np.save("data/msic1_std_test.npy", msic1_stds)
np.save("data/msic2_std_test.npy", msic2_stds)