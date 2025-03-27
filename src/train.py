import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import tools
from tensorflow.keras.utils import plot_model
import argparse

max_length = 64

# Arguments

parser = argparse.ArgumentParser(prog='train.py', description='Trains the neural network')
parser.add_argument('trainds', help="File containing training dataset in tfrecord format")
parser.add_argument('valds', help="File containing validation dataset in tfrecord format")
parser.parse_args()
args = parser.parse_args()

# Enable mixed precision
# 3070 accelerates FP16 better
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

train_ds = tools.load_tfrecord(args.trainds)
val_ds = tools.load_tfrecord(args.valds)

# Prepare for training
train_ds = train_ds.batch(128).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(128).prefetch(tf.data.AUTOTUNE)

# Initial model which has only a dense network, and uses the two extra features
v0_shape_max_length = 64
v0_input_shape = (v0_shape_max_length, 1)
v0_input_layer = tf.keras.Input(shape=v0_input_shape)
v0_first_two_items = v0_input_layer[:, :2, :]
v0_x = layers.Masking(mask_value=-1.0)(v0_first_two_items)
v0_x = layers.Flatten()(v0_x)
v0_x = layers.Dense(2, activation='relu')(v0_x)
v0_output_layer = layers.Dense(1, activation='sigmoid')(v0_x)
model_baseline = models.Model(inputs=v0_input_layer, outputs=v0_output_layer)

# Second model with two GRU layers
v1_shape_max_length = 64
v1_input_shape = (v1_shape_max_length, 1)
v1_input_layer = tf.keras.Input(shape=v1_input_shape)
v1_first_two_items = v1_input_layer[:, :2, :]
v1_flattened_two_items = layers.Flatten()(v1_first_two_items)
v1_remaining_items = v1_input_layer[:, 2:, :]
v1_x = layers.Masking(mask_value=-1.0)(v1_remaining_items)
v1_x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(v1_x)
v1_x = layers.Dropout(0.1)(v1_x)
v1_x = layers.Bidirectional(layers.GRU(64))(v1_x)
v1_x = layers.Dropout(0.1)(v1_x)
v1_x = layers.Dense(64, activation='relu', kernel_regularizer='l2')(v1_x)
v1_flattened_rnn_output = layers.Flatten()(v1_x)
v1_combined_output = layers.Concatenate()([v1_flattened_rnn_output, v1_flattened_two_items])
v1_output_layer = layers.Dense(1, activation='sigmoid')(v1_combined_output) 
model_gru = models.Model(inputs=v1_input_layer, outputs=v1_output_layer)

# Third model based on a transformer
v2_input_shape = (64, 1) 
v2_embedding_dim = 64
v2_num_heads = 4
v2_key_dim = 64
v2_ff_dim = 64
v2_dropout_rate = 0.1
v2_num_classes = 1
v2_input_layer = layers.Input(shape=v2_input_shape)
v2_first_two_items = v2_input_layer[:, :2, :]
v2_flattened_two_items = layers.Flatten()(v2_first_two_items)
v2_remaining_items = v2_input_layer[:, 2:, :]
v2_masked_inputs = layers.Masking(mask_value=-1.0)(v2_remaining_items)
v2_x = layers.Dense(v2_embedding_dim)(v2_masked_inputs)
v2_pos_encodings = tools.positional_encoding(v2_input_shape[0]-2, v2_embedding_dim)
v2_x += v2_pos_encodings
v2_attention_output = layers.MultiHeadAttention(num_heads=v2_num_heads,key_dim=v2_key_dim,dropout=v2_dropout_rate)(v2_x, v2_x)
v2_x = layers.Add()([v2_x, v2_attention_output])
v2_x = layers.LayerNormalization()(v2_x)
v2_ffn_output = layers.Dense(v2_ff_dim, activation="relu")(v2_x)
v2_ffn_output = layers.Dropout(v2_dropout_rate)(v2_ffn_output)
v2_ffn_output = layers.Dense(v2_embedding_dim)(v2_ffn_output)
v2_x = layers.Add()([v2_x, v2_ffn_output])
v2_x = layers.LayerNormalization()(v2_x)
v2_x = layers.GlobalAveragePooling1D()(v2_x)
v2_flattened_output = layers.Flatten()(v2_x)
v2_combined_output = layers.Concatenate()([v2_flattened_output, v2_flattened_two_items])
v2_outputs = layers.Dense(v2_num_classes, activation="sigmoid")(v2_combined_output)
model_transformer = models.Model(inputs=v2_input_layer, outputs=v2_outputs)

# Select the model
model = model_gru

# Display a model summary
model.summary()

# Plot the model
plot_model(model, to_file=('dolus.png'), show_shapes=True, show_trainable=True)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# For larger jobs, snapshot the best performing model
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="snapshot.keras", 
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[cp_callback])

# Evaluate the model
loss, accuracy = model.evaluate(val_ds)
print(f"Validation Accuracy: {accuracy:.2f}")

# Save the model
model.save("dolus.keras")

# Save accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy_plot.png')  # Save the plot to a file
plt.close()  # Close the plot to free up memory

# Save loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss_plot.png')  # Save the plot to a file
plt.close()  # Close the plot to free up memory

