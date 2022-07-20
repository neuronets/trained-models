#!/usr/bin/env python3
from pathlib import Path
import json
import click
import nibabel as nib
from nibabel.processing import conform, resample_from_to
from nobrainer.prediction import predict_by_estimator
from nobrainer.volume import standardize
import numpy as np

_models_dir = Path(__file__).resolve().parent 

_models = {
    'bwn': _models_dir / 'weights',
    'bwn_multi': _models_dir / 'weights',
    'bvwn_multi_prior': _models_dir / 'weights',
}


@click.command()
@click.argument('infiles', nargs=-1)
@click.argument('outprefix')
@click.option('-m', '--model', type=click.Choice(_models.keys()), default="bwn_multi", required=True, help='Model to use for prediction.')
@click.option('-n', '--n-samples', type=int, default=1, help='Number of samples to predict.')
@click.option('-b', '--batch-size', type=int, default=8, help='Batch size during prediction.')
@click.option('--save-variance', is_flag=True, help='Save volume with variance across `n-samples` predictions.')
@click.option('--save-entropy', is_flag=True, help='Save volume of entropy values.')
@click.option('--overwrite', type=click.Choice(['yes', 'skip'], case_sensitive=False), help='Overwrite existing output or skip')
@click.option('--atlocation', is_flag=True, help='Save output in the same location as input')

def predict(*, infiles, outprefix, model, n_samples, batch_size, save_variance, save_entropy, overwrite, atlocation):
    """Predict labels from features using a kwyk trained model.
    The predictions are saved to OUTPREFIX_* with the same extension as the input file.
    If you encounter out-of-memory issues, use a lower batch size value.
    """
    
    savedmodel_path = str(_models[model])
    for infile in infiles:
        _predict(infile, outprefix, savedmodel_path, n_samples, batch_size, save_variance, save_entropy, overwrite, atlocation)


def _predict(infile, outprefix, model_path, n_samples, batch_size, save_variance, save_entropy, overwrite, atlocation):
    _orig_infile = infile
    
    # Are there other neuroimaging file extensions with multiple periods?
    if infile.lower().endswith('.nii.gz'):
        outfile_ext = '.nii.gz'
    else:
        outfile_ext = Path(infile).suffix
    outfile_stem = outprefix
    
    if atlocation:
        outfile_stem = Path(infile).parent / outfile_stem

    outfile_means = "{}_means{}".format(outfile_stem, outfile_ext)
    outfile_variance = "{}_variance{}".format(outfile_stem, outfile_ext)
    outfile_entropy = "{}_entropy{}".format(outfile_stem, outfile_ext)
    outfile_uncertainty = "{}_uncertainty{}".format(outfile_stem, '.json')

    for ff in [outfile_means, outfile_variance, outfile_entropy, outfile_uncertainty]:
        if Path(ff).exists():
            if overwrite == "skip":
                return
            elif overwrite == "yes":
                pass
            else:
                raise FileExistsError("file exists: {}".format(ff))

    required_shape = (256, 256, 256)
    block_shape = (32, 32, 32)
    
    img = nib.load(infile)
    ndim = len(img.shape)
    if ndim != 3:
        raise ValueError("Input volume must have three dimensions but got {}.".format(ndim))
    if img.shape != required_shape:
        #tmp = tempfile.NamedTemporaryFile(suffix='.nii.gz')
        tmp = Path("./tmp.nii.gz")
        "conformed_tmp.nii.gz"
        print("++ Conforming volume to 1mm^3 voxels and size 256x256x256.")
        _conform(infile, tmp)
        infile = tmp
        need_reslice = True
    else:
        tmp = None
        need_reslice = False
        
    if save_variance and n_samples ==1:
        raise Exception("To calculate the variance, number of samples should be more than 1.",
                        "Please select a number bigger than 1 with '-n' or '--n_samples' argument or ",
                        "remove the --save_variance flag.")
     
    print("++ Running forward pass of model.")
    outputs = predict_by_estimator(infile, 
                                   model_path,
                                   block_shape = block_shape,
                                   batch_size = batch_size,
                                   normalizer = standardize,
                                   n_samples = n_samples,
                                   return_variance = save_variance,
                                   return_entropy = save_entropy,
                                              )
       
    # Delete temporary file.
    if tmp is not None:
        tmp.unlink()
    
    include_variance = (n_samples > 1) and (save_variance)
    if include_variance and save_entropy:
        means, variance, entropy = outputs
    elif not include_variance and save_entropy:
        means, entropy = outputs
        variance = None
    elif include_variance and not save_entropy:
        means, variance = outputs
        entropy = None
    elif not include_variance and not save_entropy:
        means = outputs
        variance = None
        entropy = None
        
    outfile_means_orig = "{}_means_orig{}".format(outfile_stem, outfile_ext)
    outfile_variance_orig = "{}_variance_orig{}".format(outfile_stem, outfile_ext)
    outfile_entropy_orig = "{}_entropy_orig{}".format(outfile_stem, outfile_ext)

    print("++ Saving results.")
    data = np.round(means.get_fdata()).astype(np.uint8)
    means = nib.Nifti1Image(data, header=means.header, affine=means.affine)
    means.header.set_data_dtype(np.uint8)
    nib.save(means, outfile_means)
    if need_reslice:    
        rs_means = _reslice(means, _orig_infile)
        nib.save(rs_means, outfile_means_orig)
        
    if save_variance and variance is not None:
        nib.save(variance, outfile_variance)
        if need_reslice:
            rs_variance = _reslice(variance, _orig_infile)
            nib.save(rs_variance, outfile_variance_orig)
    if save_entropy:
        nib.save(entropy, outfile_entropy)
        if need_reslice:
            rs_entropy = _reslice(entropy, _orig_infile)
            nib.save(rs_entropy, outfile_entropy_orig)
        uncertainty = np.mean(np.ma.masked_where(data==0, entropy.get_fdata()))
        average_uncertainty = {"uncertainty": uncertainty}
        with open(outfile_uncertainty, "w") as fp:
            json.dump(average_uncertainty, fp, indent=4)


def _conform(input_path,output_path):
    """
    Conform volume using nibabel to the shape of (256,256,256).
    
    """
    input_volume = nib.load(input_path)
    output = conform(input_volume)
    nib.save(output,output_path)
    
def _reslice(conformed_volume, orig_path, labels = False):
    reference = nib.load(orig_path)
    """reslice the output to the original volume shape and returns a resampled image object."""
    return resample_from_to(conformed_volume, reference, mode = 'nearest')


if __name__ == '__main__':
    predict()
