# Nobrainer models

This repository contains pre-trained models for 3D image processing. The models were trained using the [_Nobrainer_](https://github.com/kaczmarj/nobrainer) framework, which wraps TensorFlow/Keras. Please see the [releases](https://github.com/kaczmarj/nobrainer-models/releases) for model weights.

These models can be used for their original purpose or for transfer learning on a new task. For example, a pre-trained brain extraction network can be trained on a tumor-labeling task.

## Brain extraction

### [3D U-Net](https://github.com/kaczmarj/nobrainer/blob/master/nobrainer/models/unet.py)

This model achieved a median Dice score of 0.97, mean of 0.96, minimum of 0.91, and maximum of 0.98 on a validation dataset of 99 T1-weighted brain scans and their corresponding binarized FreeSurfer segmentations (public and private sources). This model should be agnostic to orientation and can predict the brainmask for a volume of size 256x256x256 in approximately three seconds.

This model was trained for five epochs on a dataset of 10,000 T1-weighted brain scans comprised of both public and private data. The ground truth labels were binarized FreeSurfer segmentations (i.e., binarized `aparc+aseg.mgz`). All volumes were size 256x256x256 and had 1mm isotropic voxel sizes. Because the full volumes could not fit into memory during training, each volume was separated into non-overlapping blocks of size 128x128x128, and training was performed on blocks in batches of two. The Jaccard loss function was used. During training, approximately 50% of the volumes were augmented with random rigid transformations. That is to say, 50% of the data was rotated and translated randomly in three dimensions (the transformation for pairs of features and labels was the same). The augmented features were interpolated linearly, and the augmented labels were interpolated with nearest neighbor. Each T1-weighted volume was standard scored (i.e., Z-scored) prior to training.

![Predicted brain mask on T1-weighted brain scan](/images/brain-extraction/unet-best-prediction.png)

![Predicted brain mask on T1-weighted brain scan with motion](/images/brain-extraction/unet-worst-prediction.png)
