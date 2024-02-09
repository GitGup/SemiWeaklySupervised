from common import *
import os
#from models import compileSemiWeakly #recursive issue?
from data import load_data
import sys
import time
import argparse
from utils import send_slack_message, send_slack_plot
    
qq = "qq"
noise = False
start_time = time.time()

noise_dims = 0
#load all necessary data/files
#"model_qq_opt2"
model_name = "model_qq_opt2"
model = tf.keras.models.load_model(model_name)
x = load_data("data/x_array_qqq.npy", noise_dims = noise_dims)

def eval_loss_landscape_6Features(model, feature_dims, params, m1, m2, x, step):
    
    #check if loss dictionary exists, if it does load it, if not create empty one
    dir_path = os.getcwd()
    file_name = f"data/z_{params}param{m1}{m2}{feature_dims}{model_name}.npy"
    file_path = os.path.join(dir_path, file_name)
    
    if os.path.exists(file_path):
        z = np.load(file_name, allow_pickle = True).item()
    else:
        print("Dictionary doesn't exist, creating one...")
        z = {}
    #varying sigfrac, fixed mass pair
    
    losses_list = []

    epsilon = 1e-4
    
    #if we want a specific sigfrac
    #sigspace = np.logspace(-3, -1, 10)
    sigspace = [0.001, 0.1]
    
    start = 0.5
    end = 6
    step = step

    weight_list = np.arange(start, end + step, step)
    
    for sigfrac in sigspace:
        print("Signal Fraction: ", sigfrac)
        count = 0
        for w1 in weight_list:
            for w2 in weight_list:
                if count % 1000 == 0:
                    print(f"reached {w1} {w2}")
                count+=1
                #print(w1, w2)

                inputs_hold = tf.keras.Input(shape=(1,))
                simple_model = Dense(1,use_bias = False,activation='relu',kernel_initializer=tf.keras.initializers.Constant(w1))(inputs_hold)
                model3 = Model(inputs = inputs_hold, outputs = simple_model)

                inputs_hold2 = tf.keras.Input(shape=(1,))
                simple_model2 = Dense(1,use_bias = False,activation='relu',kernel_initializer=tf.keras.initializers.Constant(w2))(inputs_hold2)
                model32 = Model(inputs = inputs_hold2, outputs = simple_model2)

                inputs_hold3 = tf.keras.Input(shape=(1,))
                simple_model3 = tf.exp(Dense(1,use_bias = False,activation='linear',kernel_initializer=tf.keras.initializers.Constant(-1))(inputs_hold3))
                model33 = Model(inputs = inputs_hold3, outputs = simple_model3)

                inputs = tf.keras.Input(shape=(feature_dims,))
                inputs2 = tf.keras.layers.concatenate([inputs,model3(tf.ones_like(inputs)[:,0]),model32(tf.ones_like(inputs)[:,0])])
                #physics prior
                hidden_layer_1 = model(inputs2)
                LLR = hidden_layer_1 / (1.-hidden_layer_1 + epsilon)

                if params == 2:
                    LLR_xs = 1 + sigfrac*LLR - sigfrac
                elif params == 3:
                    LLR_xs = 1 + model33(tf.ones_like(inputs)[:,0])*LLR
                else:
                    print("Choose 2 or 3 parameters")
                ws = LLR_xs / (1.+LLR_xs)

                SemiWeakModel = Model(inputs = inputs, outputs = ws)
                SemiWeakModel.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01))
                m1 = m1
                m2 = m2
                
                #if computed this mass pair, break
                key = (sigfrac,m1,m2)
                if key in z:
                    break

                test_background = int(1/2 * len(x[0,0, qq, noise]))
                train_reference = int(1/4 *len(x[0,0, qq, noise]))
                train_data = int(1/4 * len(x[0,0, qq, noise]))
                test_signal = int(1/2*len(x[m1,m2, qq, noise]))

                #randomized signal
                random_test_signal_length = random.randint(0, test_signal - 1)
                N = int(1/4 * (len(x[0,0, qq, noise])))
                signal = x[m1, m2, qq, noise][random_test_signal_length:random_test_signal_length + int(sigfrac*N)]

                #fixed signal portion
                #signal = x[m1, m2, qq, noise][test_signal:test_signal + int(sigfrac*N)]

                x_data_ = np.concatenate([x[0,0, qq, noise][test_background:],signal])
                y_data_ = np.concatenate([np.zeros(train_reference),np.ones(train_data),np.ones(len(signal))])
                
                X_train_, X_val_, Y_train_, Y_val_ = train_test_split(x_data_, y_data_, test_size=0.5, random_state = 42)
                
                with tf.device('/GPU:0'):
                    loss = SemiWeakModel.evaluate(X_val_, Y_val_, verbose = 0)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dims", type=int, help="Number of feature dimensions")
    parser.add_argument("--parameters", type=int, help="Number of parameters")
    parser.add_argument("--m1", type=int, help="Value for m1")
    parser.add_argument("--m2", type=int, help="Value for m2")
    parser.add_argument("--step", type=float, help="Resolution of Weight Space")
    args = parser.parse_args()
    
    message = (
    "```"
    "---------- Creating Landscape With the Following Parameters ----------\n"
    f"Feature dimensions: {args.feature_dims}\n"
    f"Parameters: {args.parameters}\n"
    f"m1: {args.m1}\n"
    f"m2: {args.m2}\n"
    f"model: {model_name}\n"
    "----------------------------------------------------------------------\n"
    "```"
)
    
    send_slack_message(message)
    print(message)
    eval_loss_landscape_6Features(model, args.feature_dims, args.parameters, args.m1, args.m2, x, args.step)
    
    filename = f"data/z_{args.parameters}param{args.m1}{args.m2}{args.feature_dims}{model}.npy"
    z = np.load(filename, allow_pickle = True).item()
    sigfrac = 0.1
    elv = 60
    azim = 20
    
    create_3D_loss_manifold(sigfrac, m1, m2, z, step, elv, azim, save = True)
    loss_landscape_nofit(sigfrac, m1, m2, z, step, save = True)
    img_paths = [f"plots/landscape{float(m1)}{float(m2)}.png", f"plots/manifolds{float(m1)}{float(m2)}.png"]
    send_slack_plot(img_paths)
    send_slack_message("Done!")