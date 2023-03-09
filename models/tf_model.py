"""
http://www.tennis-data.co.uk/alldata.php  odds
"""

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, Concatenate
import numpy as np
import param_tennis as param
from preprocessing.pipeline import Dataspring
from tensorflow.keras.optimizers import Adam, SGD
__author__='Josh Lamstein'

class Net:
    def __init__(self, p):
        self.p = p

    def get_inputs(self, feats):
        inputs = {}

        for name, column in feats.items():
            dtype = column.dtype
            if dtype == object:
                dtype = tf.string
            else:
                dtype = tf.float32

            inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
        return inputs

    def preprocessing(self, inputs):

        numeric_inputs = {name: input for name, input in inputs.items()
                          if input.dtype == tf.float32}
        numeric_lst = list(numeric_inputs.values())

        x = Concatenate(name='concat_input')(numeric_lst)
        # build custom norm, norm based on training set

        # norm = preprocessing.Normalization()
        # if stage == 'train':
        #     norm.adapt(np.array(feats[numeric_inputs.keys()]))
        # all_numeric_inputs = norm(x)
        all_numeric_inputs = x
        preprocessed_inputs = all_numeric_inputs

        # for name, input in inputs.items():
        #     if input.dtype == tf.float32:
        #         continue
        #
        #     lookup = preprocessing.StringLookup(vocabulary=np.unique(feats[name]))
        #     one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())
        #
        #     x = lookup(input)
        #     x = one_hot(x)
        #     preprocessed_inputs.append(x)
        # preprocessed_inputs_cat = Concatenate()(preprocessed_inputs)
        res = tf.keras.Model(inputs=numeric_inputs, outputs=x, name='preprocess_model')
        return res

    def mlp_from_dict(self, feats):
        inputs = self.get_inputs(feats)
        act = 'relu'
        lyr = self.preprocessing(inputs)
        # inp_lst = list(inputs.values)
        x = lyr(inputs)
        x = Dense(1028, activation=act)(x)
        x = Dense(256, activation=act)(x)
        x = Dense(256, activation=act)(x)
        x = Dense(256, activation=act)(x)
        x = Dense(128, activation=act)(x)
        x = Dense(self.p.output_size, activation='sigmoid')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        if self.p.optimizer=='adam':
            optimizer = Adam(learning_rate=self.p.learning_rate)
        elif self.p.optimizer=='sgd':
            optimizer = SGD(learning_rate=self.p.learning_rate, momentum=self.p.momentum, nesterov=self.p.nesterov)
        else:
            raise ValueError(self.p.optimizer)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def mlp_from_df(self, input_size):

        inputs = self.get_inputs(feats)
        act = 'relu'
        lyr = self.preprocessing(inputs)
        # inp_lst = list(inputs.values)
        x = lyr(inputs)
        x = Dense(128, activation=act)(x)
        x = Dense(256, activation=act)(x)
        x = Dense(256, activation=act)(x)
        x = Dense(128, activation=act)(x)
        x = Dense(self.p.output_size, activation='sigmoid')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        model.compile(optimizer="Adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
        model.summary()
        return model


if __name__ == '__main__':
    csv = r'D:\Data\Sports\tennis\tennis_data\atp_database.csv'

    Dat = Dataspring(csv)
    feats_train, feats_val, feats_test = Dat.dict_to_ds_with_labels()
    nn = Net()
    model = nn.mlp_from_dict(feats_train)
