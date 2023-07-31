import oyaml
import os
from Module.utils import extract_organization_name, get_dockerfile_path, get_pull_request_description

def check_pr():
    pr_description = get_pull_request_description()
    org_name, version_folder, model_name = extract_organization_name(pr_description)

    # Check if org_name is empty or not
    if org_name:
        # Path to the model card
        model_card_path = get_dockerfile_path() + "/model_card.yaml"

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