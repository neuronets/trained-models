from Module.utils import get_dockerfile_path

if __name__ == "__main__":
    dockerfile_path = get_dockerfile_path()
    if dockerfile_path:
        print(dockerfile_path)
    else:
        print("Dockerfile not found.")