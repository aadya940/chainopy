import json
import scipy
import numpy as np
import os


# Write functions to read CSV and real Excel

# Handle Sparse Matrices differently in `save_model`
# ==================================================

# When we call the `_learn_matrix` function in the fit
# method, we initialize elements with 1e-16 (epsilon). Hence, it
# makes sense to throw away these elements that are 1e-16 (epsilon)
# when storing the matrix as JSON for less memory footprint.


def _save_model_markovchain(_object, filename: str, epsilon: float = 1e-16):
    """
    Save the model as JSON file.

    NOTE: Not to be called directly.
    """

    # Check if tpm is sparse or can be converted to sparse
    # If Yes, store it as a sparse matrix

    if _object.tpm is not None:
        # Do sparsity checks
        sparsity = (
            np.sum(np.less_equal(_object.tpm, epsilon).astype(int))
            / _object.tpm.shape[0]
        )

        # 40% elements are zero/ near zero
        if sparsity >= 0.4:
            x = np.copy(_object.tpm)

            # Replace all very small elements by zero
            _object.tpm = np.where(x <= epsilon, 0.0, x)

            _object.tpm = scipy.sparse.coo_matrix(_object.tpm)
            json_sparse = {
                "data": _object.tpm.data.tolist(),
                "row": _object.tpm.row.tolist(),
                "col": _object.tpm.col.tolist(),
                "shape": _object.tpm.shape,
            }

            attributes = {
                "tpm": json_sparse,
                "states": _object.states,
                "eigendecom": _object.eigendecom,
                "epsilon": epsilon,
            }

            if _object.eigendecom:
                attributes["eigenvalues-real"] = _object.eigenvalues.real.tolist()
                attributes["eigenvalues-imaginary"] = _object.eigenvalues.imag.tolist()

                # eigenvalues = eigenvalues.real + i * eigenvalues.imag

                attributes["eigenvectors-real"] = _object.eigenvectors.real.tolist()
                attributes["eigenvectors-imag"] = _object.eigenvectors.imag.tolist()

                # eigenvectors = eigenvectors.real + i * eigenvectors.imag

            if filename[-4:] == "json":
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(attributes, f, ensure_ascii=False, indent=4)
            else:
                raise ValueError("Filename should end with `.json`.")

        # Convert np arrays to lists since lists are JSON
        # Serializable.

        else:
            attributes = {
                "tpm": _object.tpm.tolist(),
                "states": _object.states,
                "eigendecom": _object.eigendecom,
                "epsilon": epsilon,
            }

            if _object.eigendecom:
                attributes["eigenvalues-real"] = _object.eigenvalues.real.tolist()
                attributes["eigenvalues-imaginary"] = _object.eigenvalues.imag.tolist()

                attributes["eigenvectors-real"] = _object.eigenvectors.real.tolist()
                attributes["eigenvectors-imag"] = _object.eigenvectors.imag.tolist()

            if filename[-4:] == "json":
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(attributes, f, ensure_ascii=False, indent=4)
            else:
                raise ValueError("Filename should end with `.json`.")

    else:
        raise ValueError(
            "Can't store _object of class <MarkovChain> \
                            with transition matrix"
        )


def _load_model_markovchain(filepath):
    """
    NOTE: Not to be called directly.
    """

    # If matrix is of sparse format during loading, handle it seperately

    with open(filepath, "r") as f:
        data = json.load(f)

    if (
        ("tpm" not in data)
        and ("states" not in data)
        and ("eigendecom" not in data)
        and ("epsilon" not in data)
    ):
        raise ValueError("Incorrect file contents. Enter a valid model")

    if isinstance(data["tpm"], dict):
        # Handle Sparse Matrix Cases

        _data = data["tpm"]["data"]
        row = data["tpm"]["row"]
        col = data["tpm"]["col"]
        shape = data["tpm"]["shape"]

        sparse_matrix = scipy.sparse.coo_matrix((_data, (row, col)), shape=shape)

        transition_matrix = sparse_matrix.toarray()
        states = data["states"]
        eigendecom = data["eigendecom"]
        epsilon = data["epsilon"]

        if "eigenvalues-real" in data.keys():
            _eigenvalues = np.array(data["eigenvalues-real"]) + np.array(
                data["eigenvalues-imaginary"], np.complex_
            )

            _eigenvectors = np.array(data["eigenvectors-real"]) + np.array(
                data["eigenvectors-imag"], np.complex_
            )

            return [
                transition_matrix,
                states,
                eigendecom,
                _eigenvalues,
                _eigenvectors,
                epsilon,
            ]  # 6

        return [transition_matrix, states, eigendecom, epsilon]  # 4

    elif isinstance(data["tpm"], list):
        # Handle Non - Sparse Matrix Cases

        transition_matrix = np.array(data["tpm"])
        states = data["states"]
        eigendecom = data["eigendecom"]

        if "eigenvalues-real" in data.keys():
            _eigenvalues = np.array(data["eigenvalues-real"]) + np.array(
                data["eigenvalues-imaginary"], np.complex_
            )

            _eigenvectors = np.array(data["eigenvectors-real"]) + np.array(
                data["eigenvectors-imag"], np.complex_
            )

            return [
                transition_matrix,
                states,
                eigendecom,
                _eigenvalues,
                _eigenvectors,
            ]  # 5

        return [transition_matrix, states, eigendecom]  # 3


def _load_text(path: str):
    """
    Reads data from a text file and returns it in a
    `chainopy.MarkovChain.fit()` compatible list.

    Args
    ----
    path: str
        path of the text file

    Returns
    -------
    list: contains all the words in the text file
    """
    if (not os.path.exists(path)) or (not path.endswith(".txt")):
        raise ValueError(
            f"Enter a Valid path. {path} is an invalid or is not a text file."
        )

    _corpus = []

    with open(path, "r") as f:
        for i in f.readlines():
            words = i.split(" ")
            _corpus.extend(words)

        return _corpus
