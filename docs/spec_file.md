# The spec.yaml ReadMe Guide

For a template please refer to docs/spec.yaml

The following things should be included in your spec.yaml file

1. The singularity and docker image paths:
    1. **For Docker:**
        1. Please create an image with your specified dockerfile. You can do so by running the following command:
            
            ```powershell
            docker build -t <image_name> <path_to_dockerfile_directory>
            ```
            
        2. Now that you have created an image you can publish it to a repository:
            
            To push a Docker image to a repository, you need to follow these steps:
            
            1. Log in to the repository: Use the **`docker login`** command to log in to the container registry where you want to push the image. Replace **`<repository>`** with the URL of the registry. You will be prompted to provide your username and password or access token.
                
                ```
                docker login <repository>
                ```
                
            2. Tag the image: Before pushing, you need to tag the local image with the repository URL and desired tag. Use the **`docker tag`** command with the following syntax:
                
                ```
                docker tag <image_name> <repository>/<image_name>:<tag>
                ```
                
                Replace **`<image_name>`** with the name of your local image and **`<repository>`** with the URL of the container registry. You can specify a tag for the image, or it will default to **`latest`**.
                
                For example:
                
                ```
                docker tag myimage <repository>/myimage:latest
                ```
                
            3. Push the image: Once the image is tagged, you can push it to the repository using the **`docker push`** command:
                
                ```
                docker push <repository>/<image_name>:<tag>
                ```
                
                Replace **`<repository>`**, **`<image_name>`**, and **`<tag>`** with the appropriate values. The image will be uploaded to the specified repository.
                
                For example:
                
                ```
                docker push <repository>/myimage:latest
                ```
                
        3. Now that you have published that image you can add it to the spec.yaml under the image/docker key:
            
            ```powershell
            docker: docker://neuronets/nobrainer-zoo:deepcsr
            ```
            
    2. **For singularity:**
        1. For singularity you may create an image with your docker image. You can do so by running the following:
            1. Make sure you have Singularity installed and configured on your system.
            2. Open a terminal or command prompt.
            3. Use the following command to create a Singularity image from a Docker image:
                
                ```
                singularity build <output_image_file> docker://<docker_image>
                ```
                
                Replace **`<output_image_file>`** with the desired name and path of the Singularity image file you want to create. Replace **`<docker_image>`** with the name of the Docker image you want to convert.
                
                For example:
                
                ```
                singularity build myimage.sif docker://myrepo/myimage:latest
                ```
                
                This command will create a Singularity image file named **`myimage.sif`** from the Docker image **`myrepo/myimage:latest`**. Ensure that you have appropriate permissions and access to the Docker image.
                
            4. Wait for the Singularity image creation process to complete. This may take some time depending on the size and complexity of the Docker image.
        2. Now that you have created a singularity file, specifiy its path in the spec.yaml file under image/singularity:
            
            ```powershell
            image:
              singularity: <Path_to_the_image>
              docker: neuronets/nobrainer-zoo:deepcsr
            ```
            
2. Repo information (If none, then just include “None” in every repository variable).
3. For inference or prediction please specify the path to the [predict](http://predict.py).py file your model uses to work and actually output a prediction.
4. Under the same category of inference, create a command. For example:
    
    ```powershell
    command: f"python {MODELS_PATH}/{model}/predict.py --conf_path {infile[0]}"
    ```
    
    What this command is doing is calling python, then it will autofill with the path to the [predict.py](http://predict.py) file uploaded to the '<org>/<model_name>/<version>’ directory, after this it will pass any options or arguments you have detailed inside your predict.py to accept as a standard input. In the example above it inputs a config file path that will later be used, so therefore, the flag —conf_path is used. After the flag the actual ‘infile’ or user input should be passed with {} since its an f string.
    
    As an example, your inference key in spec.yaml should look something like this:
    
    ```powershell
    inference:
        prediction_script: "trained-models/DeepCSR/deepcsr/1.0/predict.py"
        command: f"python {MODELS_PATH}/{model}/predict.py --conf_path {infile[0]} {outfile}"
    
        #### input data characteristics
        data_spec:
          infile: {n_files: 1}
    			outfile: {n_files: 1}
    ```
    
5. If you are using the template, you can remove the training keys if you are not allowing training functionalities on your model.
6. After that, complete the model information and help.
7. If you need guidance please refer to the examples provided of previous models.