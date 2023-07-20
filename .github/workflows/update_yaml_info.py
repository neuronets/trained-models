import yaml
import os

def edit_spec_yaml(path, model_name):
    spec_file = f"{path}/spec.yaml"
    
    # Check if the spec.yaml file exists
    if not os.path.isfile(spec_file):
        print(f"Error: spec.yaml file not found in {path}")
        return False

    # Read the content of the spec.yaml file
    with open(spec_file, "r") as f:
        spec_data = yaml.safe_load(f)

    # Update the container info keys
    # The key is "image"
    container_info = spec_data.get("image", {})
    container_info["singularity"] = f"nobrainer-zoo_{model_name}.sif"
    container_info["docker"] = f"neuronets/{model_name}"

    # Update the spec.yaml content with the modified data
    spec_data["image"] = container_info

    # Write the updated data back to the spec.yaml file
    with open(spec_file, "w") as f:
        yaml.dump(spec_data, f)

    return True

if __name__ == "__main__":
    path_to_model = os.environ.get("dockerfile_path")
    model_name = os.environ.get("model_folder")
    success = edit_spec_yaml(path_to_model, model_name)
    
    if success:
        print("Spec.yaml file updated successfully!")
    else:
        print("Failed to update spec.yaml file.")