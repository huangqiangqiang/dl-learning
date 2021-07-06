from matplotlib import scale
from numpy.lib.type_check import imag
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = load_model('./c_vs_d_small.h5')
model.summary()

img_path = '../../dataset/cats_and_dogs_small/test/cats/cat.1513.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
# print(img_tensor.shape)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

# plt.imshow(img_tensor[0])
# plt.show()


output_layers = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.inputs, outputs=output_layers)

activations = activation_model.predict(img_tensor)
# print(len(activations))

first_layer_activation = activations[0]
# print(first_layer_activation.shape)

# plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
# plt.show()

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

print(layer_names)

image_per_row = 16

i = 0
for layer_activation in activations:
    layer_name = layer_names[i]
    num_of_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    num_of_cols = num_of_features // image_per_row
    display_grid = np.zeros((size * num_of_cols, size * image_per_row))

    for col in range(num_of_cols):
        for row in range(image_per_row):
            channel_image = layer_activation[0, :, :, col * image_per_row + row]
            # print('---------------------------------------')
            # print( channel_image )
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,  row * size: (row + 1) * size] = channel_image
    scale = 1./size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.imshow(display_grid)
    plt.title(layer_name)
    plt.grid(False)
    plt.show()
    i += 1
