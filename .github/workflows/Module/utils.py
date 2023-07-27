import os
import oyaml
import sys

# Get the PR description
def get_pull_request_description():
    # Fetch the GitHub Pull Request event payload using GitHub API
    pr_event_url = os.environ.get('GITHUB_EVENT_PATH', None)
    if pr_event_url is None:
        print("Error: GITHUB_EVENT_PATH environment variable not set.")
        exit()

    with open(pr_event_url, 'r') as file:
        pr_event_data = oyaml.safe_load(file)

    # Extract the pull request description from the payload
    pull_request_description = pr_event_data.get(
        'pull_request', {}).get('body', None)

    if pull_request_description is None:
        print("Error: Pull Request Description not found in the event payload.")
        exit()
    else:
        return pull_request_description
    
def extract_organization_name(pull_request_description):
    try:
        data = oyaml.safe_load(pull_request_description)
        model_details = data.get('Model Details', {})

        return [model_details.get(key, None) for key in ['Organization Name', 'Model Version', 'Model Name']]
    
    except Exception as e:
        print(f"Error while parsing YAML: {e}")
        return None
    
def get_dockerfile_path():
    pull_request_description = get_pull_request_description()
    
    org_folder, version_folder, model_name = extract_organization_name(pull_request_description)

    dockerFilePath = f"{org_folder}/{model_name}/{version_folder}"

    return dockerFilePath

def get_latest_model_name():
    pull_request_description = get_pull_request_description()
    
    org_folder, version_folder, model_name = extract_organization_name(pull_request_description)
    
    return model_name