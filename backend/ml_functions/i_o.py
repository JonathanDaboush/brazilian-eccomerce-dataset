def save_pickle(obj, path):
    """Save a Python object to a pickle file."""

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved artifact: {path}")


def load_pickle(path):
    """Load a Python object from a pickle file."""

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "rb") as file:
        obj = pickle.load(file)

    print(f"Loaded artifact: {path}")
    return obj


def export_model_package(package, folder, filename):

    os.makedirs(folder, exist_ok=True)
    if not filename.endswith(".pkl"):
        filename += ".pkl"

    filepath = os.path.join(folder, filename)
    save_pickle(package, filepath)
    return filepath


def import_model_package(folder, filename):

    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model package not found: {filepath}")

    return load_pickle(filepath)
