# APMASK Project
# Q4: Gibbs sampling

"""
Hint: scipy.stats.truncnorm may be useful
It represents a continuous random variable with values that are confined within a 
specified range. This restriction makes it useful for modeling situations where certain 
values are not possible or are outside a specific range of interest.

Implement method based on Gibbs sampling to compute p(s1, s2|y)
Consider same prior distribution for s1 and s2.

1. Plot samples of the posterior distribution of the skills given y
    Beware of burn-in!

2. Transform samples into Gaussian distributions
    Implement function that uses mean and covariance of the samples to find Gaussian 
    approximation of the posterior distribution

3. Plot histogram of the samples generated after burn-in together with the fitted Gaussian
    posterior for at least 4 different numbers of samples. Report the time required to draw samples.

4. Compare prior p(s1) with Gaussian approx. of posterior p(s1|y=1)
    and compare p(s2) with p(s2|y=1).

"""

import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import time

def Gibbs2Gauss(samples, x):
    mu = np.mean(samples)
    var = np.var(samples)

    return 1/np.sqrt(2*np.pi*var)*np.exp(-0.5*((x-mu)**2/var))



# Hyperparameters
mu_prior = 25
sigma_prior = 40
beta_inv = 1 # Variance for t
y = 1  # observation

m1 = mu_prior
m2 = mu_prior

sig1 = sigma_prior
sig2 = sigma_prior

s1 = np.random.normal(m1, np.sqrt(sig1))    # Prior distributions
s2 = np.random.normal(m2, np.sqrt(sig2))


num_samples = 2000
burn_in = 300



# Initialize to 0? As time->inf the initialization effect decreases
s1_samples = np.zeros(num_samples)
s2_samples = np.zeros(num_samples)  # vanishes, but affects the burn-in period
s1_samples[0] = s1
s2_samples[0] = s2


# Gibbs sampling
start_time = time.time()
for i in range(num_samples-1):    
    t = stats.truncnorm.rvs(a=0, b=np.inf, loc=(s1_samples[i] - s2_samples[i]), scale=np.sqrt(beta_inv))  # p(t|s1, s2, y)

    A = np.array([[1, -1]])
    sigma_s = np.array([[sig1, 0], [0, sig2]])
    m_s = np.array([[m1], [m2]])

    S = np.linalg.inv(np.linalg.inv(sigma_s) + (beta_inv)**(-1) * (A.T @ A))
    m = S @ (np.linalg.inv(sigma_s) @ m_s + beta_inv**(-1) * A.T * t)

    s1, s2 = np.random.multivariate_normal(m.ravel(), S)
    s1_samples[i+1], s2_samples[i+1] = s1, s2

print(f" {round(time.time() - start_time, 4)} seconds ")

#

plt.rcParams.update({'font.size': 12})

plt.hist(s1_samples, bins=10, density=True, label='player 1')
plt.hist(s2_samples, bins=10, density=True, label='player 2')
plt.legend()
plt.title(f"Histogram of Skills for Player 1 and Player 2: y = {y}")
plt.show()

mvg_avg = 10
plt.plot(np.convolve(s1_samples, np.ones(mvg_avg)/mvg_avg, mode = 'valid'))
plt.title(f's1 samples with m1= {m1}, with moving average = {mvg_avg}')
plt.show()

plt.plot(np.convolve(s2_samples, np.ones(mvg_avg)/mvg_avg, mode = 'valid'))
plt.title(f's2 samples with m2= {m2}, with moving average = {mvg_avg}')
plt.show()

# Comparing histogram of the samples with the approximate Gaussian distribution
x = np.linspace(0, 90, 100)
gauss = Gibbs2Gauss(s1_samples[burn_in:], x)
plt.hist(s1_samples[burn_in:], bins=40, density=True, label='s1 samples')
plt.plot(x, gauss, label= 'Gaussian distribution - s1')
plt.legend()
plt.xlabel("Skill value")
plt.ylabel("Probability")
plt.title(f"s1 sampled {num_samples} times, with a burn-in of {burn_in}")
plt.show()


# Plotting for posterior distribution and the prior distribution
x = np.linspace(-40, 90, 100)
y1 = Gibbs2Gauss(s1_samples[burn_in:], x)
y2 = Gibbs2Gauss(s2_samples[burn_in:], x)
yprior = 1/np.sqrt(2*np.pi*sigma_prior)*np.exp(-0.5*((x-mu_prior)**2/sigma_prior))

plt.plot(x, y1, label='Posterior S1')
plt.plot(x, y2, label='Posterior S2') 
plt.plot(x, yprior, label='Prior distribution')
plt.xlabel("Skill value")
plt.ylabel("Probability")
plt.title("Gaussian approximations of posteriors")
plt.legend()
plt.show()


sns.scatterplot(x=s1_samples, y=s2_samples)
# Add labels and title
plt.xlabel('s1')
plt.ylabel('s2')
plt.title('Scatter Plot of s1 vs s2')
plt.show()
