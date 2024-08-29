# Define the model
# model = tf.keras.models.Sequential()  # Corrected "Seqential" to "Sequential"
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # Corrected "Flattern" to "Flatten"
# model.add(tf.keras.layers.Dense(128, activation='relu'))  # Corrected "activations" to "activation"
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# # Corrected "crossentryopy" to "crossentropy"

# # Train the model
# model.fit(x_train, y_train, epochs=5)  # Added "epochs=5" to specify the number of training epochs

# # Save the model
# model.save('handwritten.model')