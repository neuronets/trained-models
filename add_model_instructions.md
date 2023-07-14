# **README - Trained Model Deployment**

This README provides instructions for deploying a trained model using the **`nobrainer-zoo`** framework. Follow the steps below to set up your model repository and test it locally before creating a pull request.

## **Environment Setup**

1. Create a separate environment to submit your model:
    - Install **`git`**, **`git-annex`**, and **`datalad`**. If you are using a cluster environment like OpenMind, you can add them using **`module add openmind/<module>` or installing the manually**.

## **Repository Setup**

1. **Fork the main `trained-models` repository and clone it locally:**
    - You can use either **`datalad`** or **`git`** to clone the repository. It is recommended to use **`datalad`** to avoid issues later on with saving the dataset.
    - Using git: **`git clone git@github.com:<your_github>/trained-models.git`**
    - Using datalad: **`datalad clone git@github.com:<your_github>/trained-models.git`**
2. **Create the folder structure for your new model repository:**
    - Inside the forked repository, create a folder structure following this pattern: **`<org>/<model_name>/<version>`**.
    - The organization name can refer to your affiliation.
3. **Inside the ‘<org>/<model_name>/<version>’ folder path you just created you will need to add several files:**
    - **Create a folder named weights.**
        - Under this folder you will save the already trained model file checkpoint (i. e. a .pth or .h5 file)
    - **Create a file named spec.yaml (you can refer to the template: docs/spec.yaml and documentation on docs/spec_file.md):**
    - This is the file that specifies the command for your script to run either predict, train, generate, etc.
    - Various examples of spec.yaml files can be found under each models ‘<org>/<model_name>/<version>’ folder.
        1. [https://github.com/gaiborjosue/trained-models/blob/master/neuronets/ams/0.1.0/spec.yaml](https://github.com/gaiborjosue/trained-models/blob/master/neuronets/ams/0.1.0/spec.yaml)
        2. [https://github.com/gaiborjosue/trained-models/blob/master/UCL/SynthSR/1.0.0/general/spec.yaml](https://github.com/gaiborjosue/trained-models/blob/master/UCL/SynthSR/1.0.0/general/spec.yaml)
    - **Upload your predict.py file that you use to process the model and give an output.**
    - **If you are using any other feature such as train or generate, add those files as well.**
    - **Depending on your Dockerfile you can add either a requirements.txt or a requirements.yaml file including all the necessary dependencies and libraries your container needs so that the prediction can run smoothly.**
    - **Add your Dockerfile.**
4. **Now you need to upload the trained model file saved under the ‘weights’ folder to a publicly available space. For example, you can use google drive and allow everyone to access reading permissions on that file. For more details on how to do that please visit:** [https://workspacetips.io/tips/drive/share-a-google-drive-file-publicly/](https://workspacetips.io/tips/drive/share-a-google-drive-file-publicly/)
    1. After you saved the model in a publicly available space you can run the following command:
    
    ```powershell
    git annex addurl --relaxed --file <path_to_model_file> <google_drive_url>
    ```
    
5. **After running git annex on your model file, git annex should automatically create a branch on your forked repo with the details of the model storage.**
6. **Now we should save the dataset with a commit message from the root of the trained_models repository. You can do that by running the following code:**
    
    ```powershell
    datalad save -m “Added model X”
    ```
    
7. **After you saved the dataset, we can push the changes to github with datalad:**
    
    ```powershell
    datalad push --to origin
    ```
    

## Testing your model locally before creating a Pull Request

1. **If you want to test your model before creating a pull request please go to the nobrainer-zoo repo: https://github.com/neuronets/nobrainer-zoo and clone it inside an isolated environment (recommended).**
2. **Before testing anything, please ensure you installed nobrainer-zoo. For testing purposes instead of installing nobrainer-zoo normally with pip, we are going to install it in editable mode.**
    1. To do so, please activate your new environment, go to the cloned repo of nobrainer-zoo and in the parent folder run the following:
        
        ```powershell
        pip install -e .
        ```
        
        After installation, Nobrainer-zoo should be initialized. It also needs a cache folder to download some helper files based on your needs. By default, it creates a cache folder in your home directory (`~/.nobrainer`). If you do not want the cache folder in your `home` directory, you can setup a different cache location by setting the environmental variable `NOBRAINER_CACHE`. run below command to set it.
        
        ```powershell
        export NOBRAINER_CACHE=<path_to_your_cache_directory>
        ```
        
3. **After you cloned it locally and installed nobrainer-zoo go to the ‘nobrainerzoo’ folder and select cli.py**
4. **In this python file please search for the ‘model_db_url’ variable and replace its value (the current url of trained-models repo) with your forked repo. I. E.**
    
    ```
    Before:
    # adding trained_model repository
        model_db_url = "https://github.com/neuronets/trained-models"
    
    After:
    # adding trained_model repository
        model_db_url = "https://github.com/gaiborjosue/trained-models"
    ```
    
5. **After you have done this change you can test out your model by running the cli.py script.**
    1. To ensure that you can actually run your model, please run this command:
        
        ```powershell
        python cli.py ls
        ```
        
        A list of all the available models will be printed, please ensure that your model is in the list with the format ‘<org>/<model_name>/<version>’.
        
    2. After you’ve ensured that your model is capable of being used, please run the following command to test it out.
        
        ```powershell
        python cli.py predict -m <MODEL/model/version> <path-to-your-input> <path-to-your-output> --container_type <container-docker-or-singularity>
        ```
        
    
6. **If you do not see your model under “cli.py ls” list, a quick solution is manually adding the parent folder of your new model to nobrainer’s cache.** 
    - For this please copy the parent directory of your new model (<org>/<model_name>/<version> it would be the org).
    - Then go to your terminal and do the following:
        
        ```powershell
        cd ~/.nobrainer
        ```
        
        If you changed the nobrainer cache folder, please cd into that directory.
        
    - After going to nobrainer’s cache directory please go into the folder named “trained_models”
    - Inside that folder paste your new model’s parent directory folder containing your model’s required files to run.
    - Now when you repeat the cli.py ls command your model should show up under the list.
    - Now repeat the step mentioned in 6.b

### Test your model using Nobrainer-zoo

1. **The first step to test your model using nobrainer-zoo after your model ran normally with cli.py is to initialize the nobrainer-zoo by running:**
    
    ```powershell
    nobrainer-zoo init
    ```
    

1. **After initializing your nobrainer-zoo, please run this command to see if your model is on the list of available models:**
    
    ```powershell
    nobrainer-zoo ls
    ```
    
2. **If you see your model in that list, it means it is capable of running. Now you can test your model with the example command in your spec.yaml file. In example:**
    
    
    ```powershell
    nobrainer-zoo predict -m <MODEL>/<model>/<version> <PATH_TO_INPUT> <PATH_TO_OUTPUT> --container_type <DOCKER_or_SINGULARITY>
    ```
    

1. **If it worked both with cli.py and nobrainer-zoo it means your model should be ready for merging.**

## **Creating a Pull Request**

After testing your model locally, you can create a pull request with your forked repository to merge your changes into the main **`trained-models`** repository.