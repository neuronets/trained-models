import yaml
import os

def create_model_card(pr_description, output_path):
    model_card = f"""\
## Model Details:

- **Organization Name:** {pr_description.get('Model Details', {}).get('Organization Name', '')}
- **Person or Organization Developing Model:** {pr_description.get('Model Details', {}).get('Person or Organization Developing Model', '')}
- **Model Date:** {pr_description.get('Model Details', {}).get('Model Date', '')}
- **Model Version:** {pr_description.get('Model Details', {}).get('Model Version', '')}
- **Model Type:** {pr_description.get('Model Details', {}).get('Model Type', '')}
- **Information About Training Algorithms, Parameters, Fairness Constraints, or Other Applied Approaches, and Features:** {pr_description.get('Model Details', {}).get('Information About Training Algorithms, Parameters, Fairness Constraints, or Other Applied Approaches, and Features', '')}
- **Paper or Other Resource for More Information:** {pr_description.get('Model Details', {}).get('Paper or Other Resource for More Information', '')}
- **Citation Details:** {pr_description.get('Model Details', {}).get('Citation Details', '')}
- **License:** {pr_description.get('Model Details', {}).get('License', '')}
- **Where to Send Questions or Comments About the Model:** {pr_description.get('Model Details', {}).get('Where to Send Questions or Comments About the Model', '')}

## Intended Use:

- **Primary Intended Uses:** {pr_description.get('Intended Use', {}).get('Primary Intended Uses', '')}
- **Primary Intended Users:** {pr_description.get('Intended Use', {}).get('Primary Intended Users', '')}
- **Out-of-Scope Use Cases:** {pr_description.get('Intended Use', {}).get('Out-of-Scope Use Cases', '')}

## Factors:

- **Relevant Factors:** {pr_description.get('Factors', {}).get('Relevant Factors', '')}
- **Evaluation Factors:** {pr_description.get('Factors', {}).get('Evaluation Factors', '')}

## Metrics:

- **Model Performance Measures:** {pr_description.get('Metrics', {}).get('Model Performance Measures', '')}
- **Decision Thresholds:** {pr_description.get('Metrics', {}).get('Decision Thresholds', '')}
- **Variation Approaches:** {pr_description.get('Metrics', {}).get('Variation Approaches', '')}

## Evaluation Data:

- **Datasets:** {pr_description.get('Evaluation Data', {}).get('Datasets', '')}
- **Motivation:** {pr_description.get('Evaluation Data', {}).get('Motivation', '')}
- **Preprocessing:** {pr_description.get('Evaluation Data', {}).get('Preprocessing', '')}

## Training Data:

- **Datasets:** {pr_description.get('Training Data', {}).get('Datasets', '')}
- **Motivation:** {pr_description.get('Training Data', {}).get('Motivation', '')}
- **Preprocessing:** {pr_description.get('Training Data', {}).get('Preprocessing', '')}

## Quantitative Analyses:

- **Unitary Results:** {pr_description.get('Quantitative Analyses', {}).get('Unitary Results', '')}
- **Intersectional Results:** {pr_description.get('Quantitative Analyses', {}).get('Intersectional Results', '')}

## Ethical Considerations:

- **Description:** {pr_description.get('Ethical Considerations', {}).get('description', '')}

## Caveats and Recommendations:

- **Description:** {pr_description.get('Caveats and Recommendations', {}).get('description', '')}
"""

    with open(output_path, 'w') as file:
        file.write(model_card)


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

def get_dockerfile_path(model_folder):
    for root, _, files in os.walk(model_folder):
        for file in files:
            if file.lower() == "dockerfile":
                return os.path.join(root)
    return None

def extract_organization_name(pull_request_description):
    try:
        data = yaml.safe_load(pull_request_description)
        model_details = data.get('Model Details', {})
        org_name = model_details.get('Organization Name', None)
        return org_name
    except Exception as e:
        print(f"Error while parsing YAML: {e}")
        return None
    
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


if __name__ == "__main__":
    # Use your parsed PR description here
    pr_description = yaml.safe_load(get_pull_request_description())

    model_folder = get_latest_model_name()

    dockerfile_path = get_dockerfile_path(model_folder)

    output_file_path = dockerfile_path + "/model_card.md"

    create_model_card(pr_description, output_file_path)