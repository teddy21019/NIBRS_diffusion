from pathlib import Path

def read_NIBRS_file_paths(dir_path):
    # Create a Path object for the specified directory
    dir_path = Path(dir_path)
    paths = {}

    if not dir_path.is_dir():
        raise ValueError(f"Directory {dir_path} does not exist.")

    # Iterate through subdirectories in the specified directory
    for subdir in dir_path.iterdir():

        # Get a list of files in the subdirectory
        files = [file for file in subdir.glob('*') if file.name != '.DS_Store']

        if len(files) != 1:
            print(f"Subdirectory {subdir} does not contain exactly one file.")
            continue

        paths[int(subdir.name)] = files[0]

    return dict(sorted(paths.items()))

