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

#training data
x_data_qq = np.load("data/x_parametrized_data_qq.npy")
y_data_qq = np.load("data/y_parametrized_data_qq.npy")
X_train_qq, X_val_qq, Y_train_qq, Y_val_qq = train_test_split(x_data_qq, y_data_qq, test_size=0.5, random_state = 42)

pscratch_dir = "/pscratch/sd/g/gupsingh/"
os.environ["WANDB_DIR"] = pscratch_dir

config = {
    "layer_1_neurons": 1024,
    "layer_2_neurons": 512,
    "layer_3_neurons": 256,
    "output_neurons": 1,
    "activation": "relu",
    "output_activation": "sigmoid",
    "optimizer": "adam",
    "learning_rate": 0.01,
    "loss": "binary_crossentropy",
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
        return model_parametrized, history_parametrized.history
    else:
        return model_parametrized

model_parametrized, history_parametrized = train_parametrized(X_train_qq, Y_train_qq, X_val_qq, Y_val_qq, config, return_history = True)
model_parametrized.save(pscratch_dir + run_name)

SLACK_API_TOKEN = "xoxb-327063347744-6595347520051-z1j3XpfctTSv05EQCQCbHgRP"
client = WebClient(token=SLACK_API_TOKEN)
channel_id = "D05JLSUNH8V"

message = "Parametrized Training finished running!"

try:
    response = client.chat_postMessage(channel=channel_id, text=message)
    print("Message sent successfully:", response["ts"])
except SlackApiError as e:
    print("Error sending message:", e.response["error"])

wandb.finish()
print("DONE TRAINING")

