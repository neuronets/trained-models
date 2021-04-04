# Nobrainer models

This repository contains pre-trained models for 3D image processing. The models were trained using the [_Nobrainer_](https://github.com/neuronets/nobrainer) framework, which wraps TensorFlow/Keras.

These models can be used for their original purpose or for transfer learning on a new task. For example, a pre-trained brain extraction network can be trained on a tumor-labeling task.

## Models in neuronet organization

- [brainy](https://github.com/neuronets/brainy): 3D U-Net brain extraction model
- [ams](https://github.com/neuronets/ams): automated meningioma segmentation model
- [kwyk](https://github.com/neuronets/kwyk): bayesian neural network for brain parcellation and uncertainty estimation
- braingen: progressive generation of T1-weighted brain MR scans

The folder inside the model names shows the released versions of the model.

## Downloading models

This repo is a datalad dataset. To get the models you need [`datalad`](https://www.datalad.org/get_datalad.html) and [`datalad-osf`](https://pypi.org/project/datalad-osf/). First `datalad clone` the repo and then run `datalad get -s osf-storage .` to get the whole content. 

```
datalad clone https://github.com/neuronets/trained-models
cd trained-models
datalad get -s osf-storage .
```

to get a specific model you can pass the path of the model to the `datalad get`.

```
datalad get -s osf-storage neuronets/ams/0.1.0/meningioma_T1wc_128iso_v1.h5
```

```
datalad get -s neuronets/braingen
```

## Loading the models

The models can be loaded with `tf.keras.models.load_model`. 

```
import nobrainer
import tensorflow as tf

model = tf.keras.models.load_model("neuronets/kwyk/0.4.1/all_50_bvwn_multi_prior")
model.predict(...)
```

The `h5` files (bariny and ams) contain weights that can be loaded onto an instantiated architecture.

```
import nobrainer
import tensorflow as tf

model = tf.keras.models.load_model('brain-extraction-unet-128iso-model.h5')
model.predict(...)

model = nobrainer.models.unet(n_classes=1, input_shape=(128, 128, 128, 1))
model.load_weights('brain-extraction-unet-128iso-weights.h5')
model.predict(...)
```
