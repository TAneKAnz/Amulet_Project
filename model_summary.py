import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers

base_model = InceptionV3(input_shape=(299, 299, 3), include_top=False, weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False                                                                   # Freeze the weights in all layers of the CNN

model = tf.keras.models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(19, activation='softmax'))

print("="*90)
print("{:<5} {:<45} {:<25} {:<25}".format("Idx", "Layer (type)", "Output Shape", "Param #"))
print("="*90)

for i, layer in enumerate(base_model.layers):
    output_shape = layer.output_shape if hasattr(layer, 'output_shape') else '---'
    param_count = layer.count_params()
    print("{:<5} {:<45} {:<25} {:<25}".format(i, layer.name + f" ({layer.__class__.__name__})", str(output_shape), str(param_count)))

print("="*90)

model.summary()