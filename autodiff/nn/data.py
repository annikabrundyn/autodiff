import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles


def generate_data(samples, shape_type='circles', noise=0.05):
    """Generates data.

    Generates and formats data using sklearn functionality.

    Parameters
    ----------
    samples : int
        The total number of points generated.
    shape_type : str, optional, default: 'circles'
        Whether to generate data with circular or lunar shape.
    noise : float, optional, default: 0.05
        Standard deviation of Gaussian noise added to the data.

    Returns
    -------
    data : pandas.DataFrame
        A pandas dataframe containing the data and the class
        of the generated samples.

    """
    if shape_type == 'moons':
        X, Y = make_moons(n_samples=samples, noise=noise)
    elif shape_type == 'circles':
        X, Y = make_circles(n_samples=samples, noise=noise)
    else:
        raise ValueError(f"The introduced shape {shape_type} is not valid. Please use 'moons' or 'circles' ")

    data = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=Y))

    return data


