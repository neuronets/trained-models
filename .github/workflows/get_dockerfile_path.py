import os
import yaml

def extract_organization_name(pull_request_description):
    try:
        data = yaml.safe_load(pull_request_description)
        model_details = data.get('Model Details', {})
        org_name = model_details.get('Organization Name', None)
        return org_name
    except Exception as e:
        print(f"Error while parsing YAML: {e}")
        return None

def get_pull_request_description():
    # Fetch the GitHub Pull Request event payload using GitHub API
    pr_event_url = os.environ.get('GITHUB_EVENT_PATH', None)
    if pr_event_url is None:
        print("Error: GITHUB_EVENT_PATH environment variable not set.")
        exit()

    with open(pr_event_url, 'r') as file:
        pr_event_data = yaml.safe_load(file)

    # Extract the pull request description from the payload
    pull_request_description = pr_event_data.get('pull_request', {}).get('body', None)

    if pull_request_description is None:
        print("Error: Pull Request Description not found in the event payload.")
        exit()
    else:
        return pull_request_description
    
def get_latest_model_name():
    pull_request_description = get_pull_request_description()
    
    org_folder = extract_organization_name(pull_request_description)
    
    # Cd into the org folder
    os.chdir(org_folder)

    # Assign current directory to model_n
    model_n = os.getcwd()

    if model_n == 'objects':
        exit()
    else:
        return model_n
    
def get_dockerfile_path(model_folder):
    for root, _, files in os.walk(model_folder):
        for file in files:
            if file.lower() == "dockerfile":
                return os.path.join(root)
    return None


if __name__ == "__main__":
    model_folder = get_latest_model_name()

    dockerfile_path = get_dockerfile_path(model_folder)
    if dockerfile_path:
        print(dockerfile_path)
    else:
        print("Dockerfile not found.")