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
datalad get -s osf-storage neuronets/braingen
```

## Loading models for training

All models are available for re-training or transfer learning purposes except the **kwyk** model, which its weights are not available in tf2 keras format (We are working to make it available in near future). These models can be loaded with `tf.keras.models.load_model`.

```
import tensorflow as tf

model = tf.keras.models.load_model("neuronets/brainy/0.1.0/brain-extraction-unet-128iso-model.h5")
model.fit(...)
```

You can see a transfer learning example [here](https://github.com/neuronets/nobrainer/blob/master/guide/transfer_learning.ipynb), and an example of brain MRI generation using **braingen** models can be find [here](https://github.com/neuronets/nobrainer/blob/master/guide/train_generation_progressive.ipynb).


## Using models for inference

You can use [_Nobrainer_](https://github.com/neuronets/nobrainer) toolbox for inference.

```
import nobrainer
from nobrainer.volume import standardize_numpy

block_shape=(128,128,128)
batch_size = 1
image_path = "path_to_input_image_file"

model = tf.keras.models.load_model("neuronets/brainy/0.1.0/brain-extraction-unet-128iso-model.h5")
out = nobrainer.prediction.predict_from_filepath(image_path, 
                                           model,
                                           block_shape = (128,128,128),
                                           batch_size = batch_size,
                                           normalizer = standardize_numpy,
                                             )

```

For an example of inferece using kwyk model, please see this [notebook](https://github.com/neuronets/nobrainer/blob/master/guide/inference_with_kwyk_model.ipynb).


