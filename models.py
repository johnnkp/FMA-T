# https://github.com/CVxTz/music_genre_classification/blob/master/code/models.py
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    GlobalAvgPool1D,
    GRU,
    Input,
)
from tensorflow.keras.losses import mae
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops

from transformer import Encoder, Decoder


def custom_binary_accuracy(y_true, y_pred, threshold=0.5):
    threshold = math_ops.cast(threshold, y_pred.dtype)
    y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
    y_true = math_ops.cast(y_true > threshold, y_true.dtype)

    return K.mean(math_ops.equal(y_true, y_pred), axis=-1)


def custom_binary_crossentropy(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    epsilon_ = K._constant_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    output = clip_ops.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = 4 * y_true * math_ops.log(output + K.epsilon())
    bce += (1 - y_true) * math_ops.log(1 - output + K.epsilon())
    return K.sum(-bce, axis=-1)

# MHW2202: Y. Mansar model adjustment
def transformer_classifier(
    num_layers=4,
    d_model=128,
    num_heads=8,
    dff=256,
    maximum_position_encoding=2048,
    n_classes=16,
):
    inp = Input((None, d_model))

    """ modeling_tf_bart.py: shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
        ValueError: Shape must be rank 2 but is rank 3
        for '{{node tf_bart_model/model/concat}} = ConcatV2[N=2, T=DT_FLOAT, Tidx=DT_INT32]
        (tf_bart_model/model/Fill, tf_bart_model/model/strided_slice_2, tf_bart_model/model/concat/axis)'
         with input shapes: [?,1], [?,?,128], [].
         
        modeling_tf_bart.py:
        shifted_input_layers = tf.keras.layers.Concatenate()
        shifted_input_ids = shifted_input_layers([start_tokens, input_ids[:, :-1]])
        ValueError: A `Concatenate` layer requires inputs with matching shapes except for the concatenation axis.
        Received: input_shape=[(None, 1), (None, None, 128)]
                
        BigBird/core/attention.py: first_product = tf.einsum("BHQD,BHKD->BHQK", blocked_query_matrix[:, :, 0], key_layer)
        ValueError: Dimensions must be equal, but are 64 and ? for
        '{{node bert/encoder/layer_0/self/einsum_1/Einsum}} = Einsum[N=2, T=DT_FLOAT, equation="BHQD,BHKD->BHQK"]
        (bert/encoder/layer_0/self/strided_slice_3, bert/encoder/layer_0/self/dense_1/BiasAdd)'
        with input shapes: [?,12,16,64], [?,?,128,?].
        
        TFBlenderbot*Model.modeling_tf_utils.py: return tf.gather(self.weight, input_ids)
        Value passed to parameter 'indices' has DataType float32 not in list of allowed values: int32, int64
        
        canine\local_attention.py: input_shape = bert_modeling.get_shape_list(input_tensor, expected_rank=3)
        AttributeError: 'NoneType' object has no attribute 'shape'
        
        modeling_tf_roberta.py: batch_size, seq_length = input_shape
        ValueError: too many values to unpack (expected 2) """

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=maximum_position_encoding,
        # rate=0.3,
    )

    x = encoder(inp)

    x = GlobalAvgPool1D()(x)

    x = Dense(4 * n_classes, activation="selu")(x)

    x = Dropout(0.1)(x)

    out = Dense(n_classes, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)

    opt = Adam(0.00001)

    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
    )

    model.summary()

    return model

# MHW2202: transformer pretrain adjustment 
def transformer_pretrain(
    num_layers=4, d_model=128, num_heads=8, dff=256, maximum_position_encoding=2048,
):
    inp = Input((None, d_model))

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=maximum_position_encoding,
        rate=0.2, # 0.3,
    )

    x = encoder(inp)

    out = Dense(d_model, activation="linear", name="out_pretraining")(x)

    model = Model(inputs=inp, outputs=out)

    opt = Adam(0.0001)

    model.compile(optimizer=opt, loss=mae)

    model.summary()

    return model


def rnn_classifier(
    d_model=128, n_layers=2, n_classes=16,
):
    inp = Input((None, d_model))

    x = Bidirectional(GRU(d_model, return_sequences=True))(inp)

    if n_classes > 1:
        for i in range(n_layers - 1):
            x = Bidirectional(GRU(d_model, return_sequences=True))(x)

    x = Dropout(0.2)(x)

    x = GlobalAvgPool1D()(x)

    x = Dense(4 * n_classes, activation="selu")(x)

    out = Dense(n_classes, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)

    opt = Adam(0.00001)

    model.compile(
        optimizer=opt, loss=custom_binary_crossentropy, metrics=[custom_binary_accuracy]
    )

    model.summary()

    return model

# MHW2202: transformer decoder model
def transformer_decoder(
        num_layers=4,
        d_model=128,
        num_heads=8,
        dff=256,
        maximum_position_encoding=2048,
        n_classes=16,
):
    inp = Input((None, d_model))

    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=maximum_position_encoding,
        rate=0.2,
    )

    x = decoder(inp, inp)

    x = GlobalAvgPool1D()(x)

    x = Dense(4 * n_classes, activation="selu")(x)

    x = Dropout(0.1)(x)

    out = Dense(n_classes, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

if __name__ == "__main__":
    model1 = transformer_classifier()

    model2 = rnn_classifier()
