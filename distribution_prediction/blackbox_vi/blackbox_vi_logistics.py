import jax.numpy as np
import numpy as onp
from jax import grad
from jax.random import bernoulli

from distribution_prediction.blackbox_vi.utils_plots import plot_vi_logistics
from scipy.stats import multivariate_normal

def sigmoid(X: np.ndarray,
            theta: np.ndarray
            ) -> np.ndarray:
    """
    :param X: matrix of shape N * P containing a data point per row
    :param theta: matrix of parameters: shape: 1 * P or M * P
    :return: (matrix of size N * 1 or N * M) : the sigmoid function applied to every element of the matrix X.dot(theta.T)
    """
    return np.clip(1 / (1 + np.exp(- np.dot(X, theta.T))), 1e-5, 1 - 1e-5)


def kl_div(mu: np.ndarray,
           A: np.ndarray,
           sigma_prior: float
           ) -> float:
    """
    Computes the KL divergence between
    - the approximated posterior distribution N(mu, Sigma)
    - and the prior distribution on the parameters N(0, (sigma_prior ** 2) I)

    Instead of working directly with the covariance matrix Sigma, we will only deal with its Cholesky matrix A:
    It is the lower triangular matrix such that Sigma = A * A.T

    :param mu: mean of the posterior distribution approximated by variational inference
    :param A: Choleski matrix such that Sigma = A * A.T,
    where Sigma is the coveriance of the posterior distribution approximated by variational inference.
    :param sigma_prior: standard deviation of the prior on the parameters. We put the following prior on the parameters:
    N(mean=0, variance=(sigma_prior**2) I)
    :return: the value of the KL divergence
    """


    mu.reshape(2,-1)
    d = mu.shape[1]

    theta_post_cov = A @ np.transpose(A)
    theta_prior_cov = (sigma_prior ** 2) * np.eye(d)

    t1 = np.log(np.linalg.det(theta_prior_cov)/np.linalg.det(theta_post_cov)) - d
    t2 = np.trace(np.linalg.inv(theta_prior_cov)@theta_post_cov)
    t3 = mu @ np.linalg.inv(theta_prior_cov) @ np.transpose(mu)

    kl = 0.5*(t1 + t2 + t3)

    return kl[0][0]



def expected_log_likelihood(mu: np.ndarray,
                            A: np.ndarray,
                            epsilon: np.ndarray,
                            X: np.ndarray,
                            y: np.ndarray
                            ) -> float:
    """
    :param mu: mean of the posterior distribution approximated by variational inference
    :param A: Choleski matrix such that Sigma = A * A.T,
    where Sigma is the coveriance of the posterior distribution approximated by variational inference.
    :param epsilon: The samples from N(0, I) that will be used to generate samples from
    the approximated posterior N(mu, Sigma) by using the matrix A and the vector mu
    :param X: data points of shape (N, 2) where N is the number of data points. There is one data point per row
    :param y: column vector of shape (N, 1) indicating the class of p. for each point, y_i = 0 or 1.
    In addition, y_i = 1 is equivalent to "x_i is in C_1"
    :return: The expected log-likelihood. That expectation is calculated according to the approximated posterior
    N(mu, Sigma) by using the samples in epsilon.
    """
    mu.reshape(-1, 2)
    epsilon.reshape(-1,2)
    S = []
    for e in epsilon:
        theta = mu + A @ e
        p = sigmoid(X, theta)
        bernoulli_pmf = p * y + (1-p)*(1-y)
        #S.append(np.prod(bernoulli_pmf))
        S.append(np.sum(np.log(bernoulli_pmf)))
    m = np.sum(S)/len(S)
    return m



def variational_inference_logistics(X: np.ndarray,
                                    y: np.ndarray,
                                    num_samples_per_turn: int,
                                    sigma_prior: float,
                                    number_iterations: int = 1000):
    """
    This function performs a variational inference procedure.

    Here we consider that our parameters follow a normal distribution N(mu, Sigma).
    Instead of working directly with the covariance matrix Sigma, we will only deal with its Cholesky matrix A:
    It is the lower triangular matrix such that Sigma = A * A.T

    At the end of each step, it yields the following elements (in this order):
    - the new estimated mu
    - the new estimated Sigma
    - the new estimated lower triangular matrix A (verifying A * A.T = Sigma)
    - mu_grad: the gradient of the marginal likelihood bound with respect to mu
    - A_grad: the gradient of the marginal likelihood bound with respect to A
    - epsilon: The samples from N(0, I) that were used to generate the samples from N(mu, Sigma)

    :param X: data points of shape (N, 2) where N is the number of data points. There is one data point per row
    :param y: column vector of shape (N, 1) indicating the class of p. for each point, y_i = 0 or 1.
    In addition, y_i = 1 is equivalent to "x_i is in C_1"
    :param num_samples_per_turn: number of samples of parameters to use at each step of the Blackbox Variational
    Inference Algorithm
    :param sigma_prior: standard deviation of the prior on the parameters. We put the following prior on the parameters:
    N(mean=0, variance=(sigma_prior**2) I)
    :param number_iterations: number of Blackbox Variational Inference steps before stopping
    """
    P = X.shape[1]

    counter = 0
    mu = np.zeros(shape=(1, P))
    A = np.identity(P)

    # Matrix used to make sure that the elements on the diagonal of A remain superior to 1e-5 at every step
    T = onp.full_like(A, -float('inf'))
    for i in range(P):
        T[i, i] = 1e-5

    epsilon = None
    mu_grad = None
    A_grad = None


    def L(mu, A):
        kl = kl_div(mu, A, sigma_prior)
        E = expected_log_likelihood(mu, A, epsilon, X, y)
        return E - kl

    while counter < number_iterations:
        mu_old = mu
        A_old = A

        #############################
        # TODO : Complete Here for computing epsilon, mu_grad and A_grad
        #############################


        #epsilon = multivariate_normal.rvs(mean=np.zeros(shape=P), cov=sigma_prior**2*np.eye(P), size=num_samples_per_turn)
        epsilon = multivariate_normal.rvs(mean=np.zeros(shape=P), cov=np.eye(P), size=num_samples_per_turn)
        #kl = kl_div(mu_old, A_old, sigma_prior)
        #E = expected_log_likelihood(mu_old, A_old, epsilon, X, y)

        print('aaahhh')

        mu_grad = grad(L, argnums=0)(mu_old, A_old)
        A_grad = grad(L, argnums=1)(mu_old, A_old)

        # Performing a gradient descent step on A and mu
        # (we make sure that the elements on the diagonal of A remain superior to 1e-5)
        A = np.maximum(A + (1. / (10 * counter + 100.)) * np.tril(A_grad), T)
        mu = mu + (1. / (10 * counter + 100.)) * mu_grad

        counter += 1
        if counter % 1 == 0:
            # Printing the highest change in parameters at that iteration
            print(f"counter: {counter} - {onp.max((onp.linalg.norm(mu_old - mu), onp.linalg.norm(A_old - A)))}\r")

        yield mu, A.dot(A.T), A, mu_grad, A_grad, epsilon


if __name__ == '__main__':
    plot_vi_logistics(interactive=True,
                      interval_plot=1,
                      number_points_per_class=25,
                      number_iterations=1000)
