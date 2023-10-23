# Repository "Trained-Models" - Model Submission Guide

Thank you for your interest in contributing your trained model to our "Trained-Models" repository. To ensure a smooth submission process, please follow the instructions below.

## Step 1: Create an Issue
1. Navigate to the [Issues tab](https://github.com/neuronets/trained-models/issues) of the "Trained-Models" repository.

2. Click on the "New Issue" button.

3. Select the appropriate issue template:
   - If you want to add a new model, choose "Add New Model."
   - If you want to update an existing model, choose "Update a Model."

4. In the issue title, start with "New Model: " for new models or "Update Model: " for model updates.

## Step 2: Fill Out the Issue Template
Carefully fill out the issue template with the required information:

### 1. Model's Repository Path
- Provide the model's repository path in the following format: 
`<org>/<model>/<version>` (e.g., `DeepCSR/deepcsr/1.0`).

### 2. Best Model Weights
- Paste a direct download link to the model's weights (Google Drive, Github Raw, etc.). If you need to upload multiple weights, zip the weights and paste the download link here.
  Example URL structure for Google Drive: ```"https://drive.google.com/uc?id=<FILEID>&confirm=t"```

  To extract the FILEID from your google drive share url, do the following:
  1. Right click on the file you want to share.
  2. Click share
  3. Click "Get Link"
  4. Change the general access to "Anyone with the link"
  5. Copy the link
  6. Your link should look something like this (i.e) https://drive.google.com/file/d/1KaNTsBrEohhbaUxgI8qrOFpvPbo0FEQb/view?usp=share_link
  7. The FILEID is the code that goes between ```https://drive.google.com/file/d/``` and ```/view?usp=share_link```
  8. Copy that FILEID and paste it in this template: ```"https://drive.google.com/uc?id=<FILEID>&confirm=t"```
  9. In example, the final URL should look like this: ```"https://drive.google.com/uc?id=1KaNTsBrEohhbaUxgI8qrOFpvPbo0FEQb&confirm=t"```

### 3. Docker Information
- Input a GitHub URL to a folder containing the Dockerfile and any other necessary files for Docker.

### 4. Model's Python Scripts
- Provide GitHub URLs to folders/files containing Python scripts for your model. The main Python file should not be in the same folder as helper/utils scripts. Separate multiple URLs by line.

### 5. Model Card and Spec
- Share GitHub URLs for the `model_card.yaml` and `spec.yaml` files. These files must be named as specified. For more information, refer to [this documentation](https://github.com/neuronets/trained-models/blob/master/docs/spec_file.md) and [card documentation](https://github.com/neuronets/trained-models/blob/master/docs/model_card.yaml).

### 6. Sample Data
- Upload a direct download URL pointing to the sample dataset used to test the model (Google Drive, Github Raw, etc.). If you need to upload multiple datasets, zip the files and paste the download link here. Follow the URL structure if using Google Drive: ```"https://drive.google.com/uc?id=<FILEID>&confirm=t"```

### 7. Model Config (Optional)
- Provide a GitHub URL to your model's config file if you use one. This is optional but should be from GitHub.

### 8. Test Command
- Include a test command with flags for:
  - Model weights
  - Dataset
  - Config file (optional)
  - Output directory (optional)

  The paths for the above flags should be relative to the repository structure and the model's path provided in field #1.

  Example:
  
    ```python predict.py --model_checkpoint ./<org>/<model>/<version>/weights/<weights_file> --dataset ./<org>/<model>/<version>/example-data/<dataset_file> --conf_path ./<org>/<model>/<version>/config/<config_file> --output_dir .```

### 9. Acknowledge Recommendations
- Select the checkbox to confirm that you have followed all the recommendations.

### 10. Final Steps
- Before submitting the issue ensure all your github urls point to files in the main/master branch.
- Do not add any tags; our automated process will handle them.

## Step 3: Submit the Issue
After submitting the issue, our bot will:
- Automatically create a development branch with your issue number.
- Generate a draft PR linked to the issue.
- Test the model; if it fails, the issue will be tagged as "failed."
- If you need to make changes, fix the issues, update URLs in the issue, and when ready, simply append "Ready XX" in the issue title (where XX is a number incrementing with 01 each time a fix has been applied).

## Step 4: Approval and Inclusion
- If testing is successful, the issue will be tagged as "success."
- Our bot will notify you to change the PR status to "Open."
- A developer will review and approve/reject the PR.
- Approved models will be added to the "Trained-Models" repository.

Thank you for your contribution!