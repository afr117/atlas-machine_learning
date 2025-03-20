import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Concatenate

def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in DenseNet architecture.
    
    Parameters:
    - X: output from the previous layer
    - nb_filters: number of filters in X
    - growth_rate: growth rate for the dense block
    - layers: number of layers in the dense block
    
    Returns:
    - The concatenated output of each layer within the Dense Block
    - The number of filters within the concatenated outputs
    """
    he_init = tf.keras.initializers.HeNormal(seed=0)
    
    for _ in range(layers):
        # Batch Normalization + ReLU Activation
        bn1 = BatchNormalization()(X)
        act1 = Activation('relu')(bn1)
        
        # 1x1 Convolution (Bottleneck layer)
        conv1 = Conv2D(filters=4 * growth_rate, kernel_size=1, padding='same',
                        kernel_initializer=he_init, use_bias=False)(act1)
        
        # Batch Normalization + ReLU Activation
        bn2 = BatchNormalization()(conv1)
        act2 = Activation('relu')(bn2)
        
        # 3x3 Convolution
        conv2 = Conv2D(filters=growth_rate, kernel_size=3, padding='same',
                        kernel_initializer=he_init, use_bias=False)(act2)
        
        # Concatenate the input with the new feature maps
        X = Concatenate()([X, conv2])
        nb_filters += growth_rate
    
    return X, nb_filters
