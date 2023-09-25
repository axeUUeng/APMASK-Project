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
import pandas as pd



# Hyperparameters
mu_prior = 25
sigma_prior = 40
beta_inv = 1 # Variance for to



df = pd.read_csv('SerieA.csv')
all_teams = pd.concat([df['team1'], df['team2']]).unique()
team_df = pd.DataFrame({'Teams': all_teams})
team_df['Mean'] = 25.0
team_df['Variance'] = 40.0


burn_in = 200
num_samples = 2000


for index, match in df.iterrows():

    s1_samples = np.zeros(num_samples)
    s2_samples = np.zeros(num_samples) 


    team1 = team_df.loc[team_df['Teams'] == match['team1']]
    team2 = team_df.loc[team_df['Teams'] == match['team2']]

    m1 = team1['Mean'].values[0]
    m2 = team2['Mean'].values[0]
    sig1 = team1['Variance'].values[0]
    sig2 = team2['Variance'].values[0]
   
    if(match['score1'] == match['score2']):
        continue
    
    else:
        if(match['score1'] > match['score2']):
            y = 1
            
        elif(match['score1'] < match['score2']):
            y = -1
                  

        for i in range(num_samples-1):    # Gibbs sampling
            if y == 1:
                t = stats.truncnorm.rvs(a=0, b=np.inf, loc=(s1_samples[i] - s2_samples[i]), scale=np.sqrt(beta_inv))  # p(t|s1, s2, y)
            elif y == -1:
                t = stats.truncnorm.rvs(a=-np.inf, b=0 , loc=(s1_samples[i] - s2_samples[i]), scale=np.sqrt(beta_inv))


            A = np.array([[1, -1]])
            sigma_s = np.array([[sig1, 0], [0, sig2]])
            m_s = np.array([[m1],[m2]])
        
            S = np.linalg.inv(np.linalg.inv(sigma_s) + (beta_inv)**(-1) * (A.T @ A))
            m = S @ (np.linalg.inv(sigma_s) @ m_s + beta_inv**(-1) * A.T * t)
        
            s1, s2 = np.random.multivariate_normal(m.ravel(), S)
            s1_samples[i+1], s2_samples[i+1] = s1, s2
        
     

        team_df.loc[team_df['Teams'] == match['team1'], 'Mean'] = m[0, 0]
        team_df.loc[team_df['Teams'] == match['team2'], 'Mean'] = m[1, 0]

        team_df.loc[team_df['Teams'] == match['team1'], 'Variance'] = S[0, 0]
        team_df.loc[team_df['Teams'] == match['team2'], 'Variance'] = S[1, 1]




team_df_sorted = team_df.sort_values(by='Mean', ascending=False)
print(team_df_sorted)

quit()

#
plt.hist(s1_samples, bins=10, density=True, label='player 1')
plt.hist(s2_samples, bins=10, density=True, label='player 2')
plt.legend()
plt.title(f"Histogram of Skills for Player 1 and Player 2: y = {y}")
plt.show()

mvg_avg = 10
plt.plot(np.convolve(s1_samples, np.ones(mvg_avg)/mvg_avg, mode = 'valid'))
plt.title(f'S1 samples with m1= {m1}, with moving average = {mvg_avg}')
plt.show()

plt.plot(np.convolve(s2_samples, np.ones(mvg_avg)/mvg_avg, mode = 'valid'))
plt.title(f'S2 samples with m2= {m2}, with moving average = {mvg_avg}')
plt.show()

plt.hist(s1_samples[burn_in:], bins=10, density=True, label='player 1')
plt.hist(s2_samples[burn_in:], bins=10, density=True, label='player 2')
plt.legend()
plt.title(f"Histogram of Skills for Player 1 and Player 2: y = {y}, without burn_in")
plt.show()



sns.scatterplot(x=s1_samples, y=s2_samples)
# Add labels and title
plt.xlabel('s1')
plt.ylabel('s2')
plt.title('Scatter Plot of s1 vs s2')
plt.show()
