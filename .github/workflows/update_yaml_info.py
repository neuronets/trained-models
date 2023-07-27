import oyaml
import os
from Module.utils import get_dockerfile_path, get_latest_model_name

def edit_spec_yaml(path, model_name):
    spec_file = f"{path}/spec.yaml"
    
    # Check if the spec.yaml file exists
    if not os.path.isfile(spec_file):
        print(f"Error: spec.yaml file not found in {path}")
        return False

    # Read the content of the spec.yaml file
    with open(spec_file, "r") as f:
        spec_data = oyaml.safe_load(f)

    # Update the container info keys
    # The key is "image"
    container_info = spec_data.get("image", {})
    container_info["singularity"] = f"nobrainer-zoo_{model_name}.sif"
    container_info["docker"] = f"neuronets/{model_name}"

    # Update the spec.yaml content with the modified data
    spec_data["image"] = container_info

    # Write the updated data back to the spec.yaml file
    with open(spec_file, "w") as f:
        oyaml.dump(spec_data, f)

    return True

if __name__ == "__main__":
    model_folder = get_latest_model_name()
    dockerfile_path = get_dockerfile_path()

    success = edit_spec_yaml(dockerfile_path, model_folder)
    
    if success:
        print("Spec.yaml file updated successfully!")
    else:
        print("Failed to update spec.yaml file.")
        print("Failed to update spec.yaml file.")