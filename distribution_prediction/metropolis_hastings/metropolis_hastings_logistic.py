import numpy as np


from distribution_prediction.metropolis_hastings.utils_plots import plot_metropolis_hastings_logistics


def get_log_upper_proba_distribution(X: np.ndarray,
                                     y: np.ndarray,
                                     theta: np.ndarray,
                                     sigma_prior: float
                                     ) -> float:
    """
    This functions evaluates log( p_1(theta | X, y) ) where:
     - p_1 = Z * p
     - p is the posterior distribution
     - p_1 is easy to calculate

    You may use the sigmoid function in the file utils.py
    BE CAREFUL: be sure to reshape theta before passing it as an argument to the function sigma!!

    :param X: data points of shape (N, 2) where N is the number of data points, and 2 is the number of components for
    each data point x. Each row of X represents one data point.
    :param y: column vector of shape (N, 1) indicating the class of p. for each point, y_i = 0 or 1.
    In addition, y_i = 1 is equivalent to "x_i is in C_1"
    :param theta: parameters at which we evaluate p_1. It is a numpy array (row vector) of shape (2,).
    :param sigma_prior: standard deviation of the prior on the parameters
    :return: log( p_1(theta | X, y) )
    """
    # TODO


def metropolis_hastings(X: np.ndarray,
                        y: np.ndarray,
                        number_expected_samples: int,
                        sigma_proposal_density: float = 1.):
    """
    Performs a Metropolis Hastings procedure.
    This function is a generator. After each step, it should yield a tuple containing the following elements
    (in this order):
    -  is_sample_accepted (type: bool) which indicates if the last sample from the proposal density has been accepted
    -  np.array(list_samples): numpy array of size (S, 2) where S represents the total number of previously accepted
    samples, and 2 is the number of components in theta in this logistic regression task.
    -  newly_sampled_theta: numpy array of size ()
    -  u (type: float): last random number used for deciding if the newly_sampled_theta should be accepted or not.


    :param X: data points of shape (N, 2) where N is the number of data points. There is one data point per row
    :param y: column vector of shape (N, 1) indicating the class of p. for each point, y_i = 0 or 1.
    In addition, y_i = 1 is equivalent to "x_i is in C_1"
    :param number_expected_samples: Number of samples expected from the Metropolis Hastings procedure
    :param sigma_proposal_density: Standard deviation of the proposal density.
    We consider that the proposal density corresponds to a multivariate normal distribution, with:
    - mean = null vector
    - covariance matrix = (sigma_proposal_density ** 2) identity matrix
    """

    # ----- These are some the variables you should manipulate in the main loop of that function ----------
    list_samples = []  # Every newly_sampled_theta  which is accepted should be added to the list of samples

    newly_sampled_theta = None  # Last sampled parameters (from the proposal density q)

    # Last sampled parameters (from the proposal density q) which were accepted
    # according to the Metropolis-Hastings criterion
    last_accepted_theta = np.zeros(X.shape[1]).reshape(1, -1)

    is_sample_accepted = False  # Should be True if and only if the last sample has been accepted

    u = np.random.rand()  # Random number used for deciding if newly_sampled_theta should be accepted or not

    # -------------------------------------------------------------------------------------------------

    while len(list_samples) < number_expected_samples:
        #########################
        # TODO : Complete Here
        #########################

        yield is_sample_accepted, np.array(list_samples), newly_sampled_theta, u


def get_predictions(X_star: np.ndarray,
                    array_samples_theta: np.ndarray
                    ) -> np.ndarray:
    """
    :param X_star: numpy array of shape (N, 2)
    :param array_samples_theta: np array of shape (N, 2)
    :return: estimated predictions at each point in X_star: p(y_star, a column vector, shape=(N, 1)
    """
    # TODO


if __name__ == '__main__':
    plot_metropolis_hastings_logistics(1000, interactive=True)