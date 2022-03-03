import nobrainer
import tensorflow as tf
# TODO: use pathlib instead of os
from pathlib import Path
import os
# TODO: argparse can be replaced by click or sys
import argparse
import yaml


def main(config):
    """
    Fine Tune the AMS model
    """
    # Set parameters
    n_classes = config['n_classes']
    batch_size = config['dataset_train']['batch_size']
    v = config['dataset_train']['volume_shape']
    b = config['dataset_train']['block_shape']
    volume_shape = (v, v, v)
    block_shape = (b, b, b)
    n_epochs = config['train']['epoch']
    n_train = config['dataset_train']['n_train'] #ToDo: it is required when using user data
    n_eval = config['dataset_test']['n_test'] #ToDo: it is required when using user data


    if config.get("data_train_pattern") and config.get("data_valid_pattern"):
        data_train_pattern = config["data_train_pattern"]
        data_evaluate_pattern = config["data_valid_pattern"]
        if (data_train_pattern.split(".")[-1] not in ["tfrec", "tfrecord"])\
                or (data_evaluate_pattern.split(".")[-1] not in ["tfrec", "tfrecord"]):
            # TODO: write tfrecords from csv file given by user
            raise ValueError("can't use non-tfrecord format data."
                             "convert your data in the form of tfrecords with"
                             "'nobrainer.tfrecord.write'")
    else: # using sample data if no patterns provided by the user
        # checking sample_data from the config file
        if config.get("sample_data") != 'sample_MGH':
            raise ValueError(f"only sample_MGH can be used as sample_data, "
                             f"but {config.get('sample_data')} provided")
        #Load sample Data--- inputs and labels 
        csv_of_filepaths = nobrainer.utils.get_data()
        filepaths = nobrainer.io.read_csv(csv_of_filepaths)
        train_paths = filepaths[:9]
        evaluate_paths = filepaths[9:]
        
        invalid = nobrainer.io.verify_features_labels(train_paths, num_parallel_calls=2)
        assert not invalid
        invalid = nobrainer.io.verify_features_labels(evaluate_paths)
        assert not invalid
        
        # data directory is exists because we bind it to container
        data_dir = Path(__file__).resolve().parents[2] / "data"
        #current_directory = os.getcwd()
        #final_directory = os.path.join(current_directory, r'data')
        # if not os.path.exists(final_directory):
        #    os.makedirs(final_directory)
        
        nobrainer.tfrecord.write(
            features_labels=train_paths,
            filename_template= str(data_dir / "data-train_shard-{shard:03d}.tfrec"),
            #filename_template='data/data-train_shard-{shard:03d}.tfrec',
            examples_per_shard=3)
        
        data_train_pattern = str(data_dir / "data-train_shard-*.tfrec")
        
        nobrainer.tfrecord.write(
            features_labels=evaluate_paths,
            filename_template= str(data_dir / 'data-evaluate_shard-{shard:03d}.tfrec'),
            examples_per_shard=1)
        
        data_evaluate_pattern = str(data_dir / "data-evaluate_shard-*.tfrec")
       
    # Create and Load Datasets for training and validation
    dataset_train = nobrainer.dataset.get_dataset(
        file_pattern = data_train_pattern,
        n_classes = n_classes,
        batch_size = batch_size,
        volume_shape = volume_shape,
        block_shape = block_shape,
        n_epochs = n_epochs,
        augment = config['dataset_train']['augment'],
        shuffle_buffer_size = config['dataset_train']['shuffle_buffer_size'],
        num_parallel_calls = config['dataset_train']['num_parallel_calls'],
    )
    
    dataset_evaluate = nobrainer.dataset.get_dataset(
        file_pattern = data_evaluate_pattern,
        n_classes = n_classes,
        batch_size = batch_size,
        volume_shape = volume_shape,
        block_shape = block_shape,
        n_epochs = 1,
        augment = config['dataset_test']['augment'],
        shuffle_buffer_size = config['dataset_test']['shuffle_buffer_size'],
        num_parallel_calls = config['dataset_test']['num_parallel_calls'],
    )
    
    # TODO: Add multi gpu training option
    # Compile model
    model = nobrainer.models.unet(n_classes=n_classes, 
                                  input_shape=(*block_shape, 1),
                                  batchnorm = config['network']['batchnorm'],)
    if config['path']['pretrained_model']: 
        model.load_weights(config['path']['pretrained_model'])
    optimizer = tf.keras.optimizers.Adam(learning_rate = config['train']['lr'])
    model.compile(optimizer=optimizer,
                  loss= eval(config['train']['loss']),
                  metrics= [eval(x) for x in config['train']['metrics']],)
    
    # Training Model
    steps_per_epoch = nobrainer.dataset.get_steps_per_epoch(
        n_volumes=n_train,
        volume_shape=volume_shape,
        block_shape=block_shape,
        batch_size=batch_size)
    
    print("number of steps per training epoch:", steps_per_epoch)
    
    validation_steps = nobrainer.dataset.get_steps_per_epoch(
        n_volumes=n_eval,
        volume_shape=volume_shape,
        block_shape=block_shape,
        batch_size=batch_size)
    
    print("number of steps per validation epoch:", validation_steps)
    # TODO: implement callbacks
    # callbacks = []
    # if check_point_path:
    #     cpk_call_back = tf.keras.callbacks.ModelCheckpoint(check_point_path)
    #     callbacks.append(cpk_call_back)
        
    # history = model.fit(
    model.fit(
            dataset_train,
            epochs= n_epochs,
            steps_per_epoch=steps_per_epoch, 
            validation_data=dataset_evaluate, 
            validation_steps=validation_steps,
            #callbacks=callbacks,
            )
    
    # TODO: save the training history
    # if save_history:
    #     current_directory = os.getcwd()
    #     file_name = os.pathjoin(current_directory,f"{save_history}.json")
    #     with open(file_name,"w") as file:
    #         json.dump(history, file)
            
    #save model
    save_path = os.path.join(config['path']['save_model'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.save_weights(os.path.join(save_path,'weights_ams_unet.hdf5' ))
    print("Model is saved at {}".format(os.path.join(save_path,'weights_ams_unet.hdf5' )))
    
    # TODO: Add loading a pretrained model for transfer learning 
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, help='Path to config YAML file.')
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    main(config)
