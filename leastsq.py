import numpy as np
from scipy.optimize import leastsq
import pylab as plt

N = 1000 # number of data points
t = np.linspace(0, 4*np.pi, N)
data = 3.0*np.sin(t+0.001) + 0.5 + np.random.randn(N) # create artificial data with noise ndarray


data_guess = [np.mean(data), 3*np.std(data)/(2**0.5), 0, 1]  # std, freq, phase, mean
def get_sin_model(guess, t, data):
    # This functions models
    guess_std = guess[0]
    guess_freq = guess[1]
    guess_phase = guess[2]
    guess_mean = guess[3]

    # # we'll use this to plot our first estimate. This might already be good enough for you
    # data_first_guess = guess_std * np.sin(guess_freq * t + guess_phase) + guess_mean
    # optimize_func = lambda x: x[0] * np.sin(x[1] * t + x[2]) + x[3] - data
    # est_std, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_std, guess_freq, guess_phase, guess_mean])[0]
    # recreate the fitted curve using the optimized parameters
    # data_fit = est_std * np.sin(est_freq * t + est_phase) + est_mean

    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = guess_std * np.exp(-t) * np.sin(guess_freq * np.exp(-t) * t + guess_phase) + guess_mean
    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0] * np.exp(-t) * np.sin(x[1] * np.exp(-t) * t + x[2]) + x[3] - data
    est_std, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_std, guess_freq, guess_phase, guess_mean])[0]

    # recreate the fitted curve using the optimized parameters
    data_fit = est_std * np.sin(est_freq * t + est_phase) + est_mean

    print('fitted mean, std, phase, freq are %f %f %f %f' % (est_std, est_freq, est_phase, est_mean))
    estimation = [est_std, est_freq, est_phase, est_mean]
    # plot results
    plt.plot(data, '.')
    plt.plot(data_fit, label='after fitting')
    plt.plot(data_first_guess, label='first guess')
    plt.legend()
    plt.show()

    return estimation


get_sin_model(data_guess, t, data)