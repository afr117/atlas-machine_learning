#!/usr/bin/env python3
"""
Module defining the train_transformer function.
"""
import tensorflow as tf

# Load dependencies
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate scheduler for the Transformer.
    """
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    """
    Calculates the sparse categorical cross-entropy loss, ignoring padded tokens (0).
    """
    # The padding value (0) should not contribute to the loss.
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    
    # Use SparseCategoricalCrossentropy since 'real' labels are integers.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    """
    Calculates the accuracy, ignoring padded tokens (0).
    """
    # The padding value (0) should not contribute to the accuracy.
    accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=2), tf.int64))
    
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Creates and trains a transformer model for machine translation.

    Args:
        N (int): The number of blocks in the encoder and decoder.
        dm (int): The dimensionality of the model.
        h (int): The number of heads.
        hidden (int): The number of hidden units in the fully connected layers.
        max_len (int): The maximum number of tokens per sequence.
        batch_size (int): The batch size for training.
        epochs (int): The number of epochs to train for.

    Returns:
        The trained model (tf.keras.Model).
    """
    data = Dataset(batch_size, max_len)
    
    # Vocabulary size is vocab_size + 2 (for SOS and EOS)
    input_vocab_size = data.vocab_size + 2
    target_vocab_size = data.vocab_size + 2
    
    # Initialize the Transformer model
    transformer = Transformer(
        N, dm, h, hidden, 
        input_vocab_size, target_vocab_size, 
        max_len, drop_rate=0.1
    )
    
    # --- Optimization and Loss ---
    # Learning rate schedule
    learning_rate = CustomSchedule(dm, warmup_steps=4000)
    
    # Adam optimizer with required parameters
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    # Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    # --- Training Step ---
    @tf.function
    def train_step(inp, tar):
        """Single training step using tf.function for speed."""
        tar_inp = tar[:, :-1]  # Target input sequence (without EOS)
        tar_real = tar[:, 1:]   # Target real sequence (without SOS)
        
        encoder_mask, combined_mask, decoder_mask = create_masks(inp, tar_inp)
        
        with tf.GradientTape() as tape:
            # Model prediction
            predictions, _ = transformer(inp, tar_inp, True, encoder_mask, combined_mask, decoder_mask)
            
            # Loss calculation
            loss = loss_function(tar_real, predictions)

        # Apply gradients
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        # Update metrics
        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

    # --- Training Loop ---
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inputs, target)) in enumerate(data.data_train):
            train_step(inputs, target)
            
            # Print every 50 batches
            if batch % 50 == 0:
                print('Epoch {}, batch {}: loss {} accuracy {}'.format(
                    epoch + 1, batch, train_loss.result().numpy(), 
                    train_accuracy.result().numpy())
                )

        # Print every epoch
        print('Epoch {}: loss {} accuracy {}'.format(
            epoch + 1, train_loss.result().numpy(), 
            train_accuracy.result().numpy())
        )

    return transformer
