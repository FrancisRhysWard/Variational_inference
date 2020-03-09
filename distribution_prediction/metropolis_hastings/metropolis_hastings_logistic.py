import numpy as np
from scipy.stats import norm, bernoulli
from scipy.stats import multivariate_normal

from distribution_prediction.metropolis_hastings.utils_plots import plot_metropolis_hastings_logistics
from distribution_prediction.utils import sigmoid

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
    :param theta: parameters at which we evaluate p_1. In our example, it is a numpy array (row vector) of shape (2,).
    :param sigma_prior: standard deviation of the prior on the parameters
    :return: log( p_1(theta | X, y) )
    """

    sig = sigmoid(X, np.transpose(theta))
    likelihood = bernoulli.pmf(y, sig.reshape(y.shape))
    theta_prior_pdf = multivariate_normal.pdf(theta, mean=np.zeros(theta.shape), cov=sigma_prior*np.eye(theta.shape[0]))

    
    return np.log(np.prod(likelihood.flatten()) * theta_prior_pdf)
   
    


def metropolis_hastings(X: np.ndarray,
                        y: np.ndarray,
                        number_expected_samples: int,
                        sigma_exploration_mh: float=1,
                        sigma_prior: float=1):
    """
    Performs a Metropolis Hastings procedure.
    This function is a generator. After each step, it should yield a tuple containing the following elements
    (in this order):
    -  is_sample_accepted (type: bool) which indicates if the last sample from the proposal density has been accepted
    -  np.array(list_samples): numpy array of size (S, 2) where S represents the total number of previously accepted
    samples, and 2 is the number of components in theta in this logistic regression task.
    -  newly_sampled_theta: in this example, numpy array of size (2,)
    -  u (type: float): last random number used for deciding if the newly_sampled_theta should be accepted or not.


    :param X: data points of shape (N, 2) where N is the number of data points. There is one data point per row
    :param y: column vector of shape (N, 1) indicating the class of p. for each point, y_i = 0 or 1.
    In addition, y_i = 1 is equivalent to "x_i is in C_1"
    :param number_expected_samples: Number of samples expected from the Metropolis Hastings procedure
    :param sigma_exploration_mh: Standard deviation of the proposal density.
    We consider that the proposal density corresponds to a multivariate normal distribution, with:
    - mean = null vector
    - covariance matrix = (sigma_proposal_density ** 2) identity matrix
    :param sigma_prior: standard deviation of the prior on the parameters
    """

    # ----- These are some the variables you should manipulate in the main loop of that function ----------
    list_samples = []  # Every newly_sampled_theta  which is accepted should be added to the list of samples

    newly_sampled_theta = None  # Last sampled parameters (from the proposal density q)

    is_sample_accepted = False  # Should be True if and only if the last sample has been accepted

    u = np.random.rand()  # Random number used for deciding if newly_sampled_theta should be accepted or not

    first_theta = np.zeros(X.shape[1])

    # -------------------------------------------------------------------------------------------------

    while len(list_samples) < number_expected_samples:
        #########################
        # TODO : Complete Here
        #########################
        
        q_old = multivariate_normal(mean=first_theta, cov=sigma_exploration_mh**2 * np.eye(first_theta.shape[0]))
        newly_sampled_theta = q_old.rvs()
        q_new = multivariate_normal(mean=newly_sampled_theta, cov=sigma_exploration_mh**2 * np.eye(newly_sampled_theta.shape[0]))
        
        q_new_given_old = q_old.pdf(newly_sampled_theta)
        q_old_given_new = q_new.pdf(first_theta)
        
        p_old = np.exp(get_log_upper_proba_distribution(X, y, first_theta, sigma_prior))
        p_new = np.exp(get_log_upper_proba_distribution(X, y, newly_sampled_theta, sigma_prior))
        
        #print(q_new_given_old, q_old_given_new, p_old, p_new)

        is_sample_accepted = (((q_old_given_new*p_new)/(q_new_given_old*p_old)) >= u)
        
        if is_sample_accepted:
            list_samples.append(newly_sampled_theta)
            first_theta = newly_sampled_theta
        
        yield is_sample_accepted, np.array(list_samples), newly_sampled_theta, u
        
        u = np.random.rand()

def get_predictions(X_star: np.ndarray,
                    array_samples_theta: np.ndarray
                    ) -> np.ndarray:
    """
    :param X_star: array of data points of shape (N, 2) where N is the number of data points.
    There is one data point per row
    :param array_samples_theta: np array of shape (M, 2) where M is the number of sampled set of parameters
    generated by the Metropolis-Hastings procedure. Each row corresponds to one sampled theta.
    :return: estimated predictions at each point in X_star: p(C_1|X,y,x_star)=p(y_star=1|X,y,x_star),
    where each x_star corresponds to a row in X_star. The result should be a column vector of shape (N, 1), its i'th
    row should be equal to the prediction p(C_1|X,y,x_star_i) where x_star_i corresponds to the i'th row in X_star
    """
    #y_star = []
    
    #for x_i in X_star:
    #    likelihoods = []
    #    for theta in array_samples_theta:
    #        sig = sigmoid(x_i, np.transpose(theta))
    #    
    #    y_star.append(np.mean(likelihoods))
    #return np.array(y_star)
    return np.mean(sigmoid(X_star, array_samples_theta), axis=-1)

if __name__ == '__main__':
        
        
    plot_metropolis_hastings_logistics(num_samples=1000,
                                       interactive=True,
                                       sigma_exploration_mh=1,
                                       sigma_prior=1,
                                       number_points_per_class=25)
