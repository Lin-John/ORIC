import os
import gc
from sklearn.metrics import log_loss, mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from deepctr.models import xDeepFM, DeepFM, WDL, DCN, AutoInt


def build_model(model_type, linear_feature_columns, dnn_feature_columns, task="binary"):
    if model_type == "DeepFM":
        model = DeepFM(linear_feature_columns,
           dnn_feature_columns,
           task=task,
           dnn_hidden_units=[400, 400, 400],
           )
    elif model_type == "xDeepFM":
        model = xDeepFM(
            linear_feature_columns,
            dnn_feature_columns,
            task=task,
            dnn_hidden_units=[400, 400],
            cin_layer_size=[200, 200, 200],
        )
    elif model_type == "WDL":
        model = WDL(
            linear_feature_columns,
            dnn_feature_columns,
            task=task,
            dnn_hidden_units=[1024, 512, 256],
        )
    elif model_type == "DCN":
        model = DCN(
            linear_feature_columns,
            dnn_feature_columns,
            task=task,
            dnn_hidden_units=[1024, 1024],
            cross_num=6,
        )
    else:
        model = AutoInt(
            linear_feature_columns,
            dnn_feature_columns,
            task=task,
            dnn_hidden_units=[400, 400],
            att_embedding_size=64
        )
    return model


def combine_model(base_model, inter_model, task):
    base_part = keras.models.clone_model(base_model)
    base_part.set_weights(base_model.get_weights())
    base_part._name = "base_model"
    for layer in base_part.layers:
        layer._name = "base_" + layer.name
    base_input = base_part.input
    base_output = base_part.get_layer(base_part.layers[-2].name).output
    base_part.trainable = False

    inter_part = keras.models.clone_model(inter_model)
    inter_part.set_weights(inter_model.get_weights())
    inter_part._name = "inter_model"
    for layer in inter_part.layers:
        layer._name = "inter_" + layer.name
    inter_input = inter_part.input
    inter_output = inter_part.get_layer(inter_part.layers[-2].name).output

    x = layers.add([base_output, inter_output])
    if task == "bianry":
        x = tf.sigmoid(x)
    final_output = tf.reshape(x, (-1, 1))
    model = keras.Model(
        inputs=[base_input, inter_input],
        outputs=final_output,
    )
    return model


# def combine_model(model_type,
#         base_linear_feat,
#         base_dnn_feat,
#         inter_linear_feat,
#         inter_dnn_feat):
#     base_part = build_model(model_type, base_linear_feat, base_dnn_feat)
#     base_part._name = "base_part"
#     for layer in base_part.layers:
#         layer._name = "base_" + layer.name
#     base_input = base_part.input
#     base_output = base_part.get_layer(base_part.layers[-2].name).output
#
#     inter_part = build_model(model_type, inter_linear_feat, inter_dnn_feat)
#     inter_part._name = "inter_part"
#     for layer in inter_part.layers:
#         layer._name = "inter_" + layer.name
#     inter_input = inter_part.input
#     inter_output = inter_part.get_layer(inter_part.layers[-2].name).output
#
#     x = layers.add([base_output, inter_output])
#     x = tf.sigmoid(x)
#     final_output = tf.reshape(x, (-1, 1))
#     model = keras.Model(
#         inputs=[base_input, inter_input],
#         outputs=final_output,
#     )
#     return model


def train_model(model,
                train_data,
                valid_data,
                epochs,
                batch_size,
                patience,
                model_checkpoint_file,
                task,):
    best_valid_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        breakout = False
        for file_id, (input_batch, y_batch) in enumerate(train_data):
            print("epoch", epoch, "file", file_id)

            model.fit(input_batch,
                      y_batch,
                      shuffle=True,
                      batch_size=batch_size,
                      epochs=1,
                      verbose=2,
                      )

            valid_loss = 0
            for input_valid, y_valid in valid_data:
                pred_valid = model.predict(input_valid, batch_size=batch_size)
                if task == "binary":
                    valid_loss += log_loss(y_valid, pred_valid, eps=1e-7)
                else:
                    valid_loss += mean_squared_error(y_valid, pred_valid)
            valid_loss /= len(valid_data)

            if valid_loss < best_valid_loss:
                model.save(model_checkpoint_file)
                print(
                    "[%d-%d] model saved!. Valid loss improved from %.4f to %.4f"
                    % (epoch, file_id, best_valid_loss, valid_loss)
                )
                best_valid_loss = valid_loss
                patience_counter = 0
            else:
                if patience_counter >= patience:
                    breakout = True
                    print("Early Stopping!")
                    break
                patience_counter += 1
            gc.collect()

        if breakout:
            break
