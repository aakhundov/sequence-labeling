import numpy as np


def pad_and_concat(a, b):
    max_dim = max(a.shape[1], b.shape[1])
    return np.concatenate([
        np.pad(a, [[0, 0]] + [[0, max_dim - a.shape[1]]] + [[0, 0]] * (a.ndim - 2), "constant"),
        np.pad(b, [[0, 0]] + [[0, max_dim - b.shape[1]]] + [[0, 0]] * (b.ndim - 2), "constant")
    ])


def fetch_in_batches(session, fetches, total, feed_dict=None, progress_callback=None):
    results = []
    fetched_so_far = 0
    while fetched_so_far < total:
        fetched = session.run(fetches, feed_dict)
        arr = [f for f in fetched if f.ndim > 0][0]
        num_fetched = arr.shape[0]

        for i in range(len(fetched)):
            f = fetched[i]
            if len(results) <= i:
                results.append(
                    0.0 if f.ndim == 0 else
                    np.zeros(shape=[0] * f.ndim, dtype=f.dtype)
                )
            if f.ndim == 0:
                results[i] += f * num_fetched
            elif f.ndim == 1:
                results[i] = np.concatenate((results[i], f))
            else:
                results[i] = pad_and_concat(results[i], f)

        fetched_so_far += num_fetched
        if progress_callback is not None:
            progress_callback(fetched_so_far)

    for i in range(len(results)):
        if results[i].ndim == 0:
            results[i] /= fetched_so_far

    return results
