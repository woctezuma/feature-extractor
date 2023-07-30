import numpy as np


def save_as_numpy_file(fname, data, dtype):
    np.save(
        fname,
        np.asarray(data, dtype=dtype),
        allow_pickle=False,
        fix_imports=False,
    )
