import tensorflow_model_optimization as tfmot 
import tensorflow_datasets as tfds
import tensorflow as tf

model_name = 'resulting_model'

#Some hyper parameters
#Do not modify!
input_shape = (144,144,3)
batch_size = 512
learning_rate = 0.001
epochs = 100

#Load the Wake Vision Dataset

#The first execution will require a lot of time, it has to download the whole Wake Vision dataset from Tensorflow Datasets (https://www.tensorflow.org/datasets/catalog/wake_vision) on your machine. The next executions it will simply use the downloaded data. 

#Where to save the downloaded dataset (239.25 GiB)
data_dir = "/path/to/dataset/"

#5,760,428 images, suitable for improvements in labels (not used in this example)
#train_large_ds = tfds.load('wake_vision', split="train_large", shuffle_files=True, data_dir=data_dir)

#1,322,574 images with high quality labels 
train_quality_ds = tfds.load('wake_vision', split="train_quality", shuffle_files=True, data_dir=data_dir)

validation_ds = tfds.load('wake_vision', split="validation", shuffle_files=True, data_dir=data_dir)
test_ds = tfds.load('wake_vision', split="test", shuffle_files=True, data_dir=data_dir)

#try to improve the dataset
#for exmample load the large split and try to improve its labels
#...
#...
#...

#prepare images for training
data_preprocessing = tf.keras.Sequential([
    #resize images to desired input shape
    tf.keras.layers.Resizing(input_shape[0], input_shape[1])])
    
train_quality_ds = train_quality_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_ds = validation_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

#Fixed Architecture
#Do not modify!
inputs = tf.keras.Input(shape=input_shape)
#
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(inputs)
x = tf.keras.layers.Conv2D(16, (3,3), padding='valid', strides=(2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(8, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(24, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid', strides=(2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = tf.keras.layers.Conv2D(16, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(80, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(16, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(16, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(48, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(16, (1,1), padding='valid')(x)
# add
x = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(80, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(3,3))(x)
x = tf.keras.layers.DepthwiseConv2D((7,7),  padding='valid', strides=(2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = tf.keras.layers.Conv2D(24, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(24, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(24, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(x)
x = tf.keras.layers.DepthwiseConv2D((7,7),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(24, (1,1), padding='valid')(x)
# add
x = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid', strides=(2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = tf.keras.layers.Conv2D(40, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(240, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(3,3))(x)
x = tf.keras.layers.DepthwiseConv2D((7,7),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(40, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(160, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(40, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(200, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(40, (1,1), padding='valid')(x)
# add
x = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(200, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = tf.keras.layers.Conv2D(48, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(48, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(192, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(48, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(48, (1,1), padding='valid')(x)
# add
x = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(192, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid', strides=(2,2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(480, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(384, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = tf.keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
# add
y = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = tf.keras.layers.Conv2D(384, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(96, (1,1), padding='valid')(x)
# add
x = tf.keras.layers.Add()([x, y])
#
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(480, (1,1), padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = tf.keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU(max_value=6.0)(x)
x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = tf.keras.layers.Conv2D(160, (1,1), padding='valid')(x)
#
x = tf.keras.layers.AveragePooling2D(5)(x)
x = tf.keras.layers.Conv2D(2, (1,1), padding='valid')(x)
outputs = tf.keras.layers.Reshape((num_classes,))(x)

model = tf.keras.Model(inputs, outputs)

#compile model
#do not modify!
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

#validation based early stopping
#do not modify!
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= model_name + ".tf",
    monitor='val_accuracy',
    mode='max', save_best_only=True)

#training
#do not modify!
model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[model_checkpoint_callback])


#Post Training Quantization (PTQ)
#do not modify!
model = tf.keras.models.load_model(model_name + ".tf")

def representative_dataset():
    for data in train_ds.rebatch(1).take(150) :
        yield [tf.dtypes.cast(data[0], tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8 
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

with open(model_name + ".tflite", 'wb') as f:
    f.write(tflite_quant_model)
    
#Test quantized model
#do not modify!
interpreter = tf.lite.Interpreter(tflite_quant_model)
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

correct = 0
wrong = 0

for image, label in test_ds :
    # Check if the input type is quantized, then rescale input data to uint8
    if input['dtype'] == tf.uint8:
       input_scale, input_zero_point = input["quantization"]
       image = image / input_scale + input_zero_point
       input_data = tf.dtypes.cast(image, tf.uint8)
       interpreter.set_tensor(input['index'], input_data)
       interpreter.invoke()
       if label.numpy().argmax() == interpreter.get_tensor(output['index']).argmax() :
           correct = correct + 1
       else :
           wrong = wrong + 1
print(f"\n\nTflite model test accuracy: {correct/(correct+wrong)}\n\n")
