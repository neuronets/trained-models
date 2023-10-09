# Repository "Trained-Models" - Model Submission Guide

Thank you for your interest in contributing your trained model to our "Trained-Models" repository. To ensure a smooth submission process, please follow the instructions below.

## Step 1: Create an Issue
1. Navigate to the [Issues tab](https://github.com/neuronets/trained-models/issues) of the "Trained-Models" repository.

2. Click on the "New Issue" button.

3. Select the appropriate issue template:
   - If you want to add a new model, choose "Add New Model."
   - If you want to push a new version of an existing model, choose "Update Model."

4. In the issue title, start with "New Model: " for new models or "Update Model: " for model updates.

## Step 2: Fill Out the Issue Template
Carefully fill out the issue template with the required information:

### 1. Model's Repository Path
- Provide the model's repository path in the following format: 
`<org>/<model>/<version>` (e.g., `DeepCSR/deepcsr/1.0`).

### 2. Best Model Weights
- Paste a direct download link from Google Drive to the model's weights.
  Example URL structure: ```"https://drive.google.com/uc?id=<FILEID>&confirm=t"```

### 3. Docker Information
- Input a GitHub URL to a folder containing the Dockerfile and any other necessary files for Docker.

### 4. Model's Python Scripts
- Provide GitHub URLs to folders/files containing Python scripts for your model. The main Python file should not be in the same folder as helper/utils scripts. Separate multiple URLs by line.

### 5. Model Card and Spec
- Share GitHub URLs for the `model_card.yaml` and `spec.yaml` files. These files must be named as specified. For more information, refer to [this documentation](https://github.com/neuronets/trained-models/blob/master/docs/spec_file.md) and [card documentation](https://github.com/neuronets/trained-models/blob/master/docs/model_card.yaml).

### 6. Sample Data
- Upload a Google Drive direct download URL pointing to the sample dataset used to test the model. Follow the URL structure: ```"https://drive.google.com/uc?id=<FILEID>&confirm=t"```

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
- Before submitting the issue, auto-assign it to yourself.
- Do not add any tags; our automated process will handle them.

## Step 3: Submit the Issue
After submitting the issue, our bot will:
- Automatically create a development branch with the issue number.
- Generate a draft PR linked to the issue.
- Test the model; if it fails, the issue will be tagged as "failed."
- If you need to make changes, fix the issues, update URLs in the issue, and change the tag from "failed" to "Ready-to-test."

## Step 4: Approval and Inclusion
- If testing is successful, the issue will be tagged as "success."
- Our bot will notify you to change the PR status to "Open."
- A developer will review and approve/reject the PR.
- Approved models will be added to the "Trained-Models" repository.

Thank you for your contribution!


