import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics
from data import load_data
from models import *
import argparse
import os
from utils import send_slack_message, send_slack_plot, get_stuck_weights


#load everything required
x = load_data("/pscratch/sd/g/gupsingh/x_array_fixed_EXTRAQCD.pkl", noise_dims = 0)
x_data_qq = np.load("/pscratch/sd/g/gupsingh/x_parametrized_data_qq_extra.npy")
y_data_qq = np.load("/pscratch/sd/g/gupsingh/y_parametrized_data_qq_extra.npy")
model_path = "/pscratch/sd/g/gupsingh/lilac-glade-67"
model_qq = tf.keras.models.load_model(model_path)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

#Loop over signal injection amounts M
#For a given signal injection amount, inject events according to N ~ Poission(M)
#For a given N injected events, initialize the network with w ~ Uniform.  Do this k times.

qq = "qq"
noise = False
epsilon = 1e-4

def train_semiweak(feature_dims, m1, m2, parameters, injections, m_initializations, decay = "qq"):
    maxsicandstd1 = {}
    maxsicandstd2 = {}
    msic1_runs = []
    msic2_runs = []
    std1_runs = []
    std2_runs = []
    score1_injections_raw_runs = []
    score2_injections_raw_runs = []
    weight_list1_runs = []
    weight_list2_runs = []
    weight_list3_runs = []
    weight_list4_runs = []
    initial_weights_runs = []
    tuple_rates_semiweak = {}
    tuple_rates_weak = {}
    
    qq = decay
    test_signal = int(1/2*len(x[m1,m2, qq, noise]))

    sigspace = np.flip(np.logspace(-3,-1,10))
    for sigfrac in sigspace:
        print(f"At {sigfrac} for decay {decay}")

        initial_weights = []

        msic1_median = []
        msic2 = []
        score1_injections = []
        score2_injections = []
        weight_list1_injections = []
        weight_list2_injections = []
        weight_list3_injections = []
        weight_list4_injections = []

        print(f"---------------{sigfrac}---------------")

        # N ~ Poission(M) injected events
        for injection in range(injections):
            print(f"Injecting N = {injections - injection} more times, currently on: N = {injection}")

            #randomized signal
            random_test_signal_length = random.randint(0, test_signal - 1)
            N = int(1/4 * (len(x[0,0, qq, noise])))
            signal = x[m1, m2, qq, noise][random_test_signal_length:random_test_signal_length + int(sigfrac*N)]

            score1_kruns = []
            weight_list1_kruns = []
            weight_list2_kruns = []
            weight_list3_kruns = []
            weight_list4_kruns = []
            scoreLossdict = {}

            print(f"Ensembling {m_initializations} for Signal Fraction : {sigfrac}")
            for k in range(m_initializations):

                w1 = round(random.uniform(0.5, 6),3)
                w2 = round(random.uniform(0.5, 6),3)
                initial_weights.append((w1, w2))

                print(f"Initialization: {w1} {w2}")

                if decay == "qq":
                    model_semiweak = compileSemiWeakly(sigfrac, model_qq, feature_dims, parameters, m1, m2, w1, w2)

                if decay == "qqq":
                    model_semiweak = compileSemiWeakly3Prong(sigfrac, model_qq, model_qqq, feature_dims, parameters, m1, m2, w1, w2)

                test_background = int(1/2 * len(x[0,0, qq, noise])+1)
                train_reference = int(1/4 *len(x[0,0, qq, noise]))
                train_data = int(1/4 * len(x[0,0, qq, noise]))
                test_signal = int(1/2*len(x[m1,m2, qq, noise])+1)

                x_data_ = np.concatenate([x[0,0, qq, noise][test_background:],signal])
                y_data_ = np.concatenate([np.zeros(train_reference),np.ones(train_data),np.ones(len(signal))])

                X_train_, X_val_, Y_train_, Y_val_ = train_test_split(x_data_, y_data_, test_size=0.5, random_state = 42)

                history_semiweak3prong = model_semiweak.fit(X_train_[:,0:feature_dims], Y_train_, epochs=50,
                                                       validation_data=(X_val_[:,0:feature_dims], Y_val_),batch_size=1024, verbose = 0, callbacks = [es])
                
                print(f"m1: {m1}",f"m2: {m2}", f"w1: {model_semiweak.trainable_weights[0].numpy()[0][0]}", f"w2: {model_semiweak.trainable_weights[1].numpy()[0][0]}")

                weight_list1_kruns+=[model_semiweak.trainable_weights[0].numpy()[0][0]]
                weight_list2_kruns+=[model_semiweak.trainable_weights[1].numpy()[0][0]]
                weight_list3_kruns+=[model_semiweak.trainable_weights[2].numpy()[0][0]]
                
                if decay == "qqq":
                    weight_list4_kruns+=[model_semiweak.trainable_weights[3].numpy()[0][0]]

                scores = model_semiweak.predict(np.concatenate([x[0,0, qq, noise][0:test_background],x[m1,m2, qq, noise][0:test_signal]]),batch_size=1024)
                #per-event probability
                score1_kruns.append(scores)
                scoreLossdict[(min(history_semiweak3prong.history["loss"]))] = scores

            #kruns finished
            weight_list1_injections.append(weight_list1_kruns)
            weight_list2_injections.append(weight_list2_kruns)
            weight_list3_injections.append(weight_list3_kruns)
            if decay == "qqq":
                weight_list4_injections.append(weight_list4_kruns)
            
            #now average over the k classifiers runs
            y = np.concatenate([np.zeros(test_background),np.ones(test_signal)])
            
            #get the lowest scores from dictionary of losses and scores
            top_items = sorted(scoreLossdict.items())[:3]
            lowest_losses = [x[0] for x in top_items]
            top_scores = [scoreLossdict[loss] for loss in lowest_losses]
            fpr, tpr, _ = metrics.roc_curve(y, np.median(top_scores, axis = 0))
            tuple_rates_semiweak[(sigfrac, injection)] = (fpr, tpr)
            
            epsilon = 1e-4
            msic1_kmedian = np.max(tpr/np.sqrt(fpr+epsilon))
            score1_injections.append(score1_kruns)
            msic1_median.append(msic1_kmedian)
            print(f" --- msic1_median on injection {injection}: {msic1_kmedian} ---")
            
            #weakly supervised only inject no k classifiers (out of mass initialization loop) no averaging to be done yet
            model_CWOLA = compileCWOLA(feature_dims, m1, m2)
            history_CWOLA = model_CWOLA.fit(X_train_[:,0:feature_dims], Y_train_, epochs=20,validation_data=(X_val_[:,0:feature_dims], Y_val_),batch_size=1024, verbose = 0)
            
            scores2 = model_CWOLA.predict(np.concatenate([x[0,0, qq, noise][0:test_background],x[m1,m2, qq, noise][0:test_signal]]),batch_size=1024)
            score2_injections.append(scores2)
            
            y2 = np.concatenate([np.zeros(test_background),np.ones(test_signal)])
            fpr2, tpr2, _ = metrics.roc_curve(y2, scores2)
            msic2.append(np.max(tpr2/np.sqrt(fpr2+epsilon)))
            tuple_rates_weak[(sigfrac, injection)] = (fpr2, tpr2)
        
        msic1_runs.append(np.median(msic1_median))
        print(f" --- msic1_runs for signal fraction {sigfrac}: {np.median(msic1_median)} ---")
        std1_runs.append(np.std(msic1_median))
        print(f" --- msic1_runs_std for signal fraction {sigfrac}: {np.std(msic1_median)} ---")
        
        print(f" --- msic2_runs for signal fraction {sigfrac}: {np.median(msic2)} ---")
        std2_runs.append(np.std(msic2))
        print(f" --- msic2_runs_std for signal fraction {sigfrac}: {np.std(msic2)} ---")
        msic2_runs.append(np.median(msic2))
        
        score1_injections_raw_runs.append(score1_injections)
        score2_injections_raw_runs.append(score2_injections)
        weight_list1_runs.append(weight_list1_injections)
        weight_list2_runs.append(weight_list2_injections)
        weight_list3_runs.append(weight_list3_injections)
        if decay == "qqq":
            weight_list4_runs.append(weight_list4_injections)
        
        maxsicandstd1[sigfrac] = (np.median(msic1_median), np.std(msic1_median))
        maxsicandstd2[sigfrac] = (np.median(msic2), np.std(msic2))
        np.save(f"data/maxsicandstd1_notebook{float(m1)}{float(m2)}_{decay}.npy", maxsicandstd1)
        np.save(f"data/maxsicandstd2_notebook{float(m1)}{float(m2)}_{decay}.npy", maxsicandstd2)
        
    stuck_weights = get_stuck_weights(sigspace, injections, m_initializations, m1, m2, weight_list1_runs, weight_list2_runs, decay)
    np.save(f"data/stuck_weights{float(m1)}{float(m2)}_{decay}.npy", stuck_weights)
    np.save(f"data/tuplerates{float(m1)}{float(m2)}_{decay}.npy", tuple_rates_semiweak)
    np.save(f"data/tuplerates2{float(m1)}{float(m2)}_{decay}.npy", tuple_rates_weak)
    
    np.save(f"data/msic1_median_notebook{float(m1)}{float(m2)}_{decay}.npy", maxsicandstd1)
    np.save(f"data/msic2_median_notebook{float(m1)}{float(m2)}_{decay}.npy", maxsicandstd2)
    np.save(f"data/std1_median_notebook{float(m1)}{float(m2)}_{decay}.npy", std1_runs)
    np.save(f"data/std2_median_notebook{float(m1)}{float(m2)}_{decay}.npy", std2_runs)
    np.save(f"data/weight_list1_runs_notebook{float(m1)}{float(m2)}_{decay}.npy", weight_list1_runs)
    np.save(f"data/weight_list2_runs_notebook{float(m1)}{float(m2)}_{decay}.npy", weight_list2_runs)
    np.save(f"data/weight_list3_runs_notebook{float(m1)}{float(m2)}_{decay}.npy", weight_list3_runs)
    
    np.save(f"data/score1_injections_raw_runs{float(m1)}{float(m2)}_{decay}.npy", score1_injections_raw_runs)
    np.save(f"data/score2_injections_raw_runs{float(m1)}{float(m2)}_{decay}.npy", score2_injections_raw_runs)
    if decay == "qqq":
        np.save(f"data/weight_list4_runs_notebook{float(m1)}{float(m2)}_{decay}.npy", weight_list4_runs)
    np.save(f"data/initial_weights_runs_notebook{float(m1)}{float(m2)}_{decay}.npy", initial_weights_runs)
    
if __name__ == "__main__":
    mass_range = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dims", type=int, help="Number of feature dimensions")
    parser.add_argument("--m1", type=float, help="Value for m1", choices=mass_range)
    parser.add_argument("--m2", type=float, help="Value for m2", choices=mass_range)
    parser.add_argument("--parameters", type=int, help="Number of parameters")
    parser.add_argument("--injections", type=int, help="Number of Randomized Signal Injections")
    parser.add_argument("--m_initializations", type=int, help="Number of Mass Initializations")
    args = parser.parse_args()
    
    message = (
    "```"
    "---------- Training Semi-Weak With the Following Parameters ----------\n"
    f"Feature dimensions: {args.feature_dims}\n"
    f"Parameters: {args.parameters}\n"
    f"m1: {args.m1}\n"
    f"m2: {args.m2}\n"
    f"model: {model_path}\n"
    "----------------------------------------------------------------------\n"
    "```"
)
    send_slack_message(message)
    train_semiweak(args.feature_dims, args.m1, args.m2, args.parameters, args.injections, args.m_initializations)
    send_slack_message("Done!")