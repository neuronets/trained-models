# Trained models

This repository contains pre-trained models for 3D neuroimaging data processing. These models can be used for their original purpose or for transfer learning on a new task. For example, a pre-trained brain extraction network can be trained on a tumor-labeling task. Models are included based on `<org-name>/<model-name>/<version>/weights` structure. Some models (such as kwyk and braingen) have various types which means there was different structural chracteristic during training that lead to different trained models. Therefor, the path for these models are changes as `<org-name>/<model-name>/<version>/<model-type>/weights`
Instructions to add a model can be find [here](https://github.com/neuronets/trained-models/blob/master/add_model_instructions.md).

## Neuronets organization

These models were trained using the [_Nobrainer_](https://github.com/neuronets/nobrainer) framework, which wraps TensorFlow/Keras.

- [brainy](https://github.com/neuronets/brainy): 3D U-Net brain extraction model 
- [ams](https://github.com/neuronets/ams): automated meningioma segmentation model
- [kwyk](https://github.com/neuronets/kwyk): bayesian neural network for brain parcellation and uncertainty estimation (Tensorflow/estimator)
- braingen: progressive generation of T1-weighted brain MR scans
  
## UCL organization

- [SynthSeg](https://github.com/BBillot/SynthSeg): 3D brain MRI segmentation model (Tensorflow/keras)
- [SynthSR](https://github.com/BBillot/SynthSR): 3D brain MRI (& CT) super resolution model (Tensorflow/keras)

## DDIG Organization

- [SynthMorph](https://github.com/voxelmorph/voxelmorph): contrast agnostic registration model (Tensorflow/keras)
- [VoxelMorph](https://github.com/voxelmorph/voxelmorph): learning based registration model (Tensorflow/keras)

## Laboratory for Computational Neuroscience (lcn) Organization

- ParcNet: cortical parcellation model (pytorch)
  
## Downloading models

This repository is a datalad dataset. To get the models, you need to install [`datalad`](https://www.datalad.org/get_datalad.html) and [`datalad-osf`](https://pypi.org/project/datalad-osf/) to your environment. 

```
datalad clone https://github.com/neuronets/trained-models
cd trained-models
git-annex enableremote osf-storage
```

to download all the models,

```
datalad get -s osf-storage .
```

to get a specific model you can pass the path of the model to the `datalad get`.

```
datalad get -s osf-storage neuronets/ams/0.1.0/weights/meningioma_T1wc_128iso_v1.h5
```

```
datalad get -s osf-storage neuronets/braingen/0.1.0
```

## Using models for inference or training

You can use the [Nobrainer-zoo](https://github.com/neuronets/zoo) toolbox for inference and re-training of the models without installing any additional model dependencies.

## Loading models for training with python and tensorflow/keras

You can use `tensorflow.keras` module to load a tensorflow model.   

```
import tensorflow as tf

model = tf.keras.models.load_model("neuronets/brainy/0.1.0/brain-extraction-unet-128iso-model.h5")
model.fit(...)
```
You can see a transfer learning example [here](https://github.com/neuronets/nobrainer/blob/master/guide/transfer_learning.ipynb), and an example of brain MRI generation using **braingen** models can be find [here](https://github.com/neuronets/nobrainer/blob/master/guide/train_generation_progressive.ipynb).

All models are available for re-training or transfer learning purposes except the **kwyk** model.  The kwyk model weights are not available in a tf2 keras format (We are working to make it available in near future). The kwyk models can be loaded with `tf.saved_model.load`.

```
model = tf.saved_model.load(model_path)
predictor = model.signatures["serving_default"]
```

or you can use nobrainer [predict_by_estimator](https://github.com/neuronets/nobrainer/blob/00325e0ed03664ae2f560547df6641dafcf8a672/nobrainer/prediction.py#L492) function. check an example [here](https://github.com/neuronets/nobrainer/blob/master/guide/inference_with_kwyk_model.ipynb).
