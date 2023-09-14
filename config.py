import wandb
#initializies wandb config
if not wandb.run:
    wandb.init(
        # set the wandb project where this run will be logged
        project="Anomaly",
        group="Dedicated",
        entity='gup-singh',

        config={
            "layer_1": 256,
            "activation_1": "relu",
            "layer_2": 256,
            "activation_2": "relu",
            "layer_3": 256,
            "activation_3": "relu",
            "output_layer": 1,
            "output_activation": "sigmoid",
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metric": "accuracy",
            "epoch": 20,
            "batch_size": 1024
        }
    )

config = wandb.config