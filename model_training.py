import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics
from wandb.keras import WandbCallback
import wandb
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os
from utils import send_slack_message, send_slack_plot

#training data
x_data_qq = np.load("/pscratch/sd/g/gupsingh/x_parametrized_data_qq_fixed.npy")
y_data_qq = np.load("/pscratch/sd/g/gupsingh/y_parametrized_data_qq_fixed.npy")
X_train_qq, X_val_qq, Y_train_qq, Y_val_qq = train_test_split(x_data_qq, y_data_qq, test_size=0.5, random_state = 42)

pscratch_dir = "/pscratch/sd/g/gupsingh/"
os.environ["WANDB_DIR"] = pscratch_dir

config = {
    "layer_1_neurons": 256,
    "layer_2_neurons": 128,
    "layer_3_neurons": 64,
    "output_neurons": 1,
    "activation": "relu",
    "output_activation": "sigmoid",
    "optimizer": "adam",
    "learning_rate": 0.01,
    "loss": "MeanSquaredError",
    "epochs": 1,
    "batch_size": 1024
}

wandb.init(project="SemiWeakly", 
           group="Parametrized", 
           entity='gup-singh', 
           mode = 'online',
           config=config)

config = wandb.config
run_name = wandb.run.name
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

def train_parametrized(X_train, Y_train, X_val, Y_val, config, return_history=False):
    model_parametrized = Sequential()
    model_parametrized.add(Dense(config["layer_1_neurons"], input_dim=np.shape(X_train_qq)[1], activation=config["activation"]))
    model_parametrized.add(BatchNormalization())
    model_parametrized.add(Dense(config["layer_2_neurons"], activation=config["activation"]))
    model_parametrized.add(BatchNormalization())
    model_parametrized.add(Dense(config["layer_3_neurons"], activation=config["activation"]))
    model_parametrized.add(BatchNormalization())
    model_parametrized.add(Dense(config.output_neurons, activation=config["output_activation"]))
    model_parametrized.compile(loss=config["loss"], optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), metrics=['accuracy'])

    with tf.device('/GPU:0'):
        history_parametrized = model_parametrized.fit(X_train, Y_train, epochs=config.epochs, validation_data=(X_val, Y_val), batch_size=config.batch_size, callbacks=[es, WandbCallback()])
        
    if return_history:
        return model_parametrized, history_parametrized
    else:
        return model_parametrized

model_parametrized, history_parametrized = train_parametrized(X_train_qq, Y_train_qq, X_val_qq, Y_val_qq, config, return_history = True)
model_parametrized.save(pscratch_dir + run_name)

wandb.finish()

num_epochs_trained = len(history_parametrized.history['loss'])
val_accuracy = history_parametrized['val_accuracy']

#Diagonistic Plot
plt.figure()
plt.plot(history_parametrized['loss'], label='Training Loss')
plt.plot(history_parametrized['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
img_path = "plots/parametrized_trainingloss.png"
plt.savefig(img_path)
plt.legend()
plt.show()

send_slack_message(f"Done Training - Epochs: {num_epochs_trained} - Validation Accuracy: {val_accuracy}")
send_slack_plot(img_path)