import oyaml
import os
from modelScheme import ModelCard, ModelSpec

def create_model_card(card_path):
    # Load the infile yaml file
    with open(card_path, 'r') as file:
        data = oyaml.safe_load(file)

    # Extract the model details from the yaml file
    model_details = data.get('Model Details', {})
    model_intended_use = data.get('Intended Use', {})
    model_factors = data.get('Factors', {})
    model_metrics = data.get('Metrics', {})
    model_evaluation_data = data.get('Evaluation Data', {})
    model_training_data = data.get('Training Data', {})
    model_quantitative_analyses = data.get('Quantitative Analyses', {})
    model_ethical_considerations = data.get('Ethical Considerations', {})
    model_caveats_and_recommendations = data.get(
        'Caveats and Recommendations', {})

    # Create a ModelCard object
    model_card = ModelCard(
        organization=model_details.get('Organization', None),
        modelDate=model_details.get('Model Date', None),
        modelVersion=model_details.get('Model Version', None),
        modelType=model_details.get('Model Type', None),
        moreInformation=model_details.get('Information About the Model', {}).get(
            'Paper or Other Resource for More Information', None),
        citationDetails=model_details.get('Citation Details', None),
        contactInfo=model_details.get(
            'Where to Send Questions or Comments About the Model', None),
        primaryIntendedUses=model_intended_use.get(
            'Primary Intended Uses', None),
        primaryIntendedUsers=model_intended_use.get(
            'Primary Intended Users', None),
        outOfScopeUseCases=model_intended_use.get(
            'Out-of-Scope Use Cases', None),
        relevantFactors=model_factors.get('Relevant Factors', None),
        evaluationFactors=model_factors.get('Evaluation Factors', None),
        modelPerformanceMeasures=model_metrics.get(
            'Model Performance Measures', None),
        decisionThresholds=model_metrics.get('Decision Thresholds', None),
        variationApproaches=model_metrics.get('Variation Approaches', None),
        evaluationDatasets=model_evaluation_data.get('Datasets', None),
        evaluationMotivation=model_evaluation_data.get('Motivation', None),
        evaluationPreprocessing=model_evaluation_data.get(
            'Preprocessing', None),
        trainingDatasets=model_training_data.get('Datasets', None),
        trainingMotivation=model_training_data.get('Motivation', None),
        trainingPreprocessing=model_training_data.get('Preprocessing', None),
        unitaryResults=model_quantitative_analyses.get(
            'Unitary Results', None),
        intersectionalResults=model_quantitative_analyses.get(
            'Intersectional Results', None),
        ethicalConsiderations=model_ethical_considerations,
        caveatsAndRecommendations=model_caveats_and_recommendations
    )

    card_data = {
        'Model_details': {
            'Organization': model_card.organization,
            'Model_date': model_card.modelDate,
            'Model_version': model_card.modelVersion,
            'Model_type': model_card.modelType,
            'More_information': model_card.moreInformation,
            'Citation_details': model_card.citationDetails,
            'Contact_info': model_card.contactInfo
        },
        'Intended_use': {
            'Primary_intended_uses': model_card.primaryIntendedUses,
            'Primary_intended_users': model_card.primaryIntendedUsers,
            'Out_of_scope_use_cases': model_card.outOfScopeUseCases,

        },
        'Factors': {
            'Relevant_factors': model_card.relevantFactors,
            'Evaluation_factors': model_card.evaluationFactors,
            'Model_performance_measures': model_card.modelPerformanceMeasures,

        },
        'Metrics': {
            'Model Performance Measures': model_card.modelPerformanceMeasures,
            'Decision Thresholds': model_card.decisionThresholds,
            'Variation Approaches': model_card.variationApproaches,
        },

        'Evaluation Data': {
            'Datasets': model_card.evaluationDatasets,
            'Motivation': model_card.evaluationMotivation,
            'Preprocessing': model_card.evaluationPreprocessing

        },

        'Training Data': {
            'Datasets': model_card.trainingDatasets,
            'Motivation': model_card.trainingMotivation,
            'Preprocessing': model_card.trainingPreprocessing
        },

        'Quantitative Analyses': {
            'Unitary Results': model_card.unitaryResults,
            'Intersectional Results': model_card.intersectionalResults
        },

        'Ethical Considerations': model_card.ethicalConsiderations,
        'Caveats and Recommendations': model_card.caveatsAndRecommendations

    }

    with open(card_path, 'w') as file:
        oyaml.dump(card_data, file)

def create_spec_yaml(model_path):

    # Load the infile yaml file
    with open(model_path, 'r') as file:
        data = oyaml.safe_load(file)
    
    # Extract the model details from the yaml file
    model_image = data.get('image', {})
    model_repository = data.get('repository', {})
    model_inference = data.get('inference', {})
    model_training_data_info = data.get('training_data_info', {})

    # Create a ModelSpec object
    model_spec = ModelSpec(
        dockerImage=model_image.get('docker', None),
        singularityImage=model_image.get('singularity', None),
        repoUrl=model_repository.get("repo_url", None),
        committish=model_repository.get("committish", None),
        repoDownload=model_repository.get("repo_download", None),
        repoDownloadLocation=model_repository.get("repo_download_location", None),
        prediction_script=model_inference.get('prediction_script', None),
        command=model_inference.get('command', None),
        n_files=model_inference.get('data_spec', {}).get('infile', {}).get('n_files', None),
        on_files=model_inference.get('data_spec', {}).get('outfile', {}).get('n_files', None),
        total=model_training_data_info.get('data_number', {}).get('total', None),
        train=model_training_data_info.get('data_number', {}).get('train', None),
        evaluate=model_training_data_info.get('data_number', {}).get('evaluate', None),
        test=model_training_data_info.get('data_number', {}).get('test', None),
        male=model_training_data_info.get('biological_sex', {}).get('male', None),
        female=model_training_data_info.get('biological_sex', {}).get('female', None),
        age_histogram=model_training_data_info.get('age_histogram', None),
        race=model_training_data_info.get('race', None),
        imaging_contrast_info=model_training_data_info.get('imaging_contrast_info', None),
        dataset_sources=model_training_data_info.get('dataset_sources', None),
        number_of_sites=model_training_data_info.get('data_sites', {}).get('number_of_sites', None),
        sites=model_training_data_info.get('data_sites', {}).get('sites', None),
        scanner_models=model_training_data_info.get('scanner_models', None),
        hardware=model_training_data_info.get('hardware', None),
        input_shape=model_training_data_info.get('training_parameters', {}).get('input_shape', None),
        block_shape=model_training_data_info.get('training_parameters', {}).get('block_shape', None),
        n_classes=model_training_data_info.get('training_parameters', {}).get('n_classes', None),
        lr=model_training_data_info.get('training_parameters', {}).get('lr', None),
        n_epochs=model_training_data_info.get('training_parameters', {}).get('n_epochs', None),
        total_batch_size=model_training_data_info.get('training_parameters', {}).get('total_batch_size', None),
        number_of_gpus=model_training_data_info.get('training_parameters', {}).get('number_of_gpus', None),
        loss_function=model_training_data_info.get('training_parameters', {}).get('loss_function', None),
        metrics=model_training_data_info.get('training_parameters', {}).get('metrics', None),
        data_preprocessing=model_training_data_info.get('training_parameters', {}).get('data_preprocessing', None),
        data_augmentation=model_training_data_info.get('training_parameters', {}).get('data_augmentation', None)
    )


    spec_data = {
        'image': {
            'docker': model_spec.dockerImage,
            'singularity': model_spec.singularityImage
        },

        'repository': {
            'repo_url': model_spec.repoUrl,
            'committish': model_spec.committish,
            'repo_download': model_spec.repoDownload,
            'repo_download_location': model_spec.repoDownloadLocation
        },

        'inference': {
            'prediction_script': model_spec.prediction_script,
            'command': model_spec.command,
            'data_spec': {
                'infile': {"n_files": model_spec.n_files},
                'outfile': {"n_files": model_spec.on_files}
            }
        },

        'training_data_info': {
            'data_number': {
                'total': model_spec.total,
                'train': model_spec.train,
                'evaluate': model_spec.evaluate,
                'test': model_spec.test
            },
            'biological_sex': {
                'male': model_spec.male,
                'female': model_spec.female
            },
            'age_histogram': model_spec.age_histogram,
            'race': model_spec.race,
            'imaging_contrast_info': model_spec.imaging_contrast_info,
            'dataset_sources': model_spec.dataset_sources,
            'data_sites': {
                'number_of_sites': model_spec.number_of_sites,
                'sites': model_spec.sites
            },
            'scanner_models': model_spec.scanner_models,
            'hardware': model_spec.hardware,
            'training_parameters': {
                'input_shape': model_spec.input_shape,
                'block_shape': model_spec.block_shape,
                'n_classes': model_spec.n_classes,
                'lr': model_spec.lr,
                'n_epochs': model_spec.n_epochs,
                'total_batch_size': model_spec.total_batch_size,
                'number_of_gpus': model_spec.number_of_gpus,
                'loss_function': model_spec.loss_function,
                'metrics': model_spec.metrics,
                'data_preprocessing': model_spec.data_preprocessing,
                'data_augmentation': model_spec.data_augmentation
            }
        }
    }

    with open(model_path, 'w') as file:
        oyaml.dump(spec_data, file)

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


def get_dockerfile_path(model_folder):
    for root, _, files in os.walk(model_folder):
        for file in files:
            if file.lower() == "dockerfile":
                return os.path.join(root)
    return None


def extract_organization_name(pull_request_description):
    try:
        data = oyaml.safe_load(pull_request_description)
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
    # Get the PR description
    pr_description = oyaml.safe_load(get_pull_request_description())

    # Get the latest model name
    model_folder = get_latest_model_name()

    # Get the dockerfile path
    dockerfile_path = get_dockerfile_path(model_folder)

    # Create the model card and spec yaml file
    card_path = dockerfile_path + "/model_card.yaml"
    spec_path = dockerfile_path + "/spec.yaml"

    create_model_card(card_path)
    create_spec_yaml(spec_path)