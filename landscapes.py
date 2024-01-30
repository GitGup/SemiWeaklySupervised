from common import *
import os
from IPython.display import display, clear_output
from PIL import Image, ImageSequence
from models import compileSemiWeakly

from mpl_toolkits.mplot3d import Axes3D
from data import load_data
import sys
import time

mass_range = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]
x = load_data("x_array_qqq.npy", noise_dims = 0)

#older version keeping here just in case
# def create_loss_landscape_6Features(model, m1, m2):
    
#     Nfeatures = 6
#     #check if loss dictionary exists, if it does load it, if not create empty one
#     dir_path = os.getcwd()
#     file_name = "z_allm1m2_{}Features.npy".format(Nfeatures)
#     file_path = os.path.join(dir_path, file_name)
    
#     if os.path.exists(file_path):
#         z = np.load(file_name, allow_pickle = True).item()
#     else:
#         print("Dictionary doesn't exist, creating one...")
#         z = {}
#     #varying sigfrac, fixed mass pair
    
#     losses_list = []

#     epsilon = 1e-6
#     sig_space = np.logspace(-3, -1, 20)
#     for sig in sig_space:
#         print("Signal Fraction: ", sig)
#         #print(w1, w2)
#         for w1 in mass_range:
#             for w2 in mass_range:
#                 sigfrac = sig

#                 for l in model.layers:
#                     l.trainable=False

#                 inputs_hold = tf.keras.Input(shape=(1,))
#                 simple_model = Dense(1,use_bias = False,activation='relu',kernel_initializer=tf.keras.initializers.Constant(w1))(inputs_hold)
#                 model3 = Model(inputs = inputs_hold, outputs = simple_model)

#                 inputs_hold2 = tf.keras.Input(shape=(1,))
#                 simple_model2 = Dense(1,use_bias = False,activation='relu',kernel_initializer=tf.keras.initializers.Constant(w2))(inputs_hold2)
#                 model32 = Model(inputs = inputs_hold2, outputs = simple_model2)

#                 inputs_hold3 = tf.keras.Input(shape=(1,))
#                 simple_model3 = tf.exp(Dense(1,use_bias = False,activation='linear',kernel_initializer=tf.keras.initializers.Constant(-1))(inputs_hold3))
#                 model33 = Model(inputs = inputs_hold3, outputs = simple_model3)

#                 inputs = tf.keras.Input(shape=(Nfeatures,))
#                 inputs2 = tf.keras.layers.concatenate([inputs,model3(tf.ones_like(inputs)[:,0]),model32(tf.ones_like(inputs)[:,0])])
#                 hidden_layer_1 = model(inputs2)
#                 LLR = hidden_layer_1 / (1.-hidden_layer_1 + epsilon)
#                 LLR_xs = 1.+sigfrac*LLR - sigfrac
#                 #LLR_xs = 1.+model33(tf.ones_like(inputs)[:,0])*LLR
#                 ws = LLR_xs / (1.+LLR_xs+0.0001)
#                 model_all2 = Model(inputs = inputs, outputs = ws)
#                 model_all2.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01))

#                 m1 = m1
#                 m2 = m2
                
#                 #if computed this mass pair, break
                
#                 key = (sigfrac,m1,m2)
#                 if key in z:
#                     break

#                 test_background = int(1/2 *len(x[0,0, qq]))
#                 train_background = int(1/4 * len(x[0,0,qq]))
#                 train_data = int(1/4 * len(x[0,0,qq]))
#                 train_reference = int(1/4 * len(x[0,0,qq]))
#                 #signal
#                 test_signal_length = int(1/2*len(x[m1,m2,qq]))
#                 sig_frac = sigfrac

#                 #randomize signal events
#                 #random_test_signal_length = random.randint(0, test_signal_length - 1)
#                 N = int(1/4 * (len(x[0,0,qq])))
#                 signal = x[m1, m2,qq][test_signal_length:test_signal_length + int(sigfrac*N)]

#                 x_vals_ = np.concatenate([x[0,0,qq][test_background:],signal])
#                 y_vals_ = np.concatenate([np.zeros(train_reference),np.ones(train_data),np.ones(len(signal))])

#                 X_train_, X_val_, Y_train_, Y_val_ = train_test_split(x_vals_, y_vals_, test_size=0.5, random_state = 42)

#                 loss = model_all2.evaluate(X_val_, Y_val_, verbose = 0)
#                 losses_list.append(loss)
                
#         if key in z:
#             print("Landscape for m1 = {} ".format(m1) + "and " + "m2 = {} ".format(m2) +" already exists for " + "{}".format(sigfrac) + " signal fraction")
#         else:
#             z[sigfrac, m1, m2] = losses_list
#             losses_list = []
#             np.save(filename, z)

#newer version of create_loss_landscape
qq = "qq"
start_time = time.time()

def create_loss_landscape_6Features(feature_dims, parameters, m1, m2, x):
    
    #check if loss dictionary exists, if it does load it, if not create empty one
    dir_path = os.getcwd()
    file_name = f"z_allm1m2_{feature_dims}test.npy"
    file_path = os.path.join(dir_path, file_name)
    
    if os.path.exists(file_path):
        z = np.load(file_name, allow_pickle = True).item()
    else:
        print("Dictionary doesn't exist, creating one...")
        z = {}
    #varying sigfrac, fixed mass pair
    
    losses_list = []

    epsilon = 1e-4
    sig_space = np.logspace(-3, -1, 20)
    
    #if we want a specific sigfrac
    #sig_space = [0.1]
    
    start = 0.5
    end = 6
    step = 0.25

    weight_list = np.arange(start, end + step, step)
    
    for sig in sig_space:
        print("Signal Fraction: ", sig)
        count = 0
        for w1 in weight_list:
            for w2 in weight_list:
                if count % 1000 == 0:
                    print(f"reached {w1} {w2}")
                count+=1
                #print(w1, w2)
                sigfrac = sig

                model_semiweak = compileSemiWeakly(feature_dims, parameters, m1, m2, w1, w2)

                m1 = m1
                m2 = m2
                
                #if computed this mass pair, break
                
                key = (sigfrac,m1,m2)
                if key in z:
                    break

                test_background = int(1/2 *len(x[0,0, qq]))
                train_background = int(1/4 * len(x[0,0,qq]))
                train_data = int(1/4 * len(x[0,0,qq]))
                train_reference = int(1/4 * len(x[0,0,qq]))
                #signal
                test_signal_length = int(1/2*len(x[m1,m2,qq]))
                sig_frac = sigfrac

                #randomize signal events
                random_test_signal_length = random.randint(0, test_signal_length - 1)
                N = int(1/4 * (len(x[0,0,qq])))
                signal = x[m1, m2,qq][random_test_signal_length:random_test_signal_length + int(sigfrac*N)]

                x_data_ = np.concatenate([x[0,0,qq][test_background:],signal])
                y_data_ = np.concatenate([np.zeros(train_reference),np.ones(train_data),np.ones(len(signal))])

                X_train_, X_val_, Y_train_, Y_val_ = train_test_split(x_data_, y_data_, test_size=0.5, random_state = 42)
                
                with tf.device('/GPU:0'):
                    loss = model_semiweak.evaluate(X_val_, Y_val_, verbose = 0)
                losses_list.append(loss)
                
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 3)
        print(f"Time taken: {elapsed_time} seconds")
        if key in z:
            print("Landscape for m1 = {} ".format(m1) + "and " + "m2 = {} ".format(m2) +" already exists for " + "{}".format(sigfrac) + " signal fraction")
        else:
            z[sigfrac, m1, m2] = losses_list
            losses_list = []
            np.save(file_name, z)
end_time_total = time.time()

elapsed_time_total = round(end_time_total - start_time, 3)
print(f"Total elapsed time: {elapsed_time_total} seconds")
    
if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python example.py <feature_dims> <parameters> <m1> <m2> <x_dict>")
        sys.exit(1)

    feature_dims = sys.argv[1]
    parameters = sys.argv[2]
    m1 = sys.argv[3]
    m2 = sys.argv[4]
    create_loss_landscape_6Features(feature_dims, parameters, m1, m2, x)