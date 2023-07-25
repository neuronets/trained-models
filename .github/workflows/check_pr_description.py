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

# Get the path to the model files
def get_dockerfile_path(model_folder):
    for root, _, files in os.walk(model_folder):
        for file in files:
            if file.lower() == "dockerfile":
                return os.path.join(root)
    return None

def check_pr():
    pr_description = oyaml.safe_load(get_pull_request_description())

    # Check if the pull request description key "Organization Name:" is empty or not
    model_details = pr_description.get('Model Details', {})
    org_name = model_details.get('Organization Name', None)

    # Check if org_name is empty or not
    if org_name:
        # Check if it matches the org name in model card
        # Cd into the org folder
        os.chdir(org_name)

        # Assign current directory to model_n
        model_n = os.getcwd()

        # Path to the model card
        model_card_path = get_dockerfile_path(model_n) + "/model_card.yaml"

        # Load the model card
        with open(model_card_path, 'r') as file:
            model_card = oyaml.safe_load(file)
        
        # Check if the org name in model card matches the org name in the pull request description
        if model_card.get("Model Details", {}).get("Organization", None) == org_name:
            print(f"The 'Organization Name:' key is not empty. Value: {org_name}")
    else:
        print("The 'Organization Name:' key is empty. PR cannot be accepted.")
        exit(1)

if __name__ == '__main__':
    check_pr()