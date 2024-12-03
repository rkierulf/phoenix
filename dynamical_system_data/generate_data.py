import numpy as np
import os
import matplotlib.pyplot as plt

np.random.seed(42)
num_time_points = 10
variances = [0.05, 0.10, 0.15, 0.2, 0.25]

#Dynamical system from section 5.1 of https://arxiv.org/pdf/2306.07803
def calc_next(x, x_prev1, x_prev2, x_prev3, variance):
    x0 = x[0] + 0.95 * np.sqrt(2) * x_prev1[0] - 0.9025 * x_prev2[0]
    x1 = x[1] + x_prev2[0] ** 2
    x2 = x[2] - 0.4 * x_prev3[0]
    x3 = x[3] - 0.5 * x_prev2[0] ** 2 + 0.5 * np.sqrt(2) * x_prev1[3] + 0.25 * np.sqrt(2) * x_prev1[4]
    x4 = x[4] - 0.5 * np.sqrt(2) * x_prev1[3] + 0.5 * np.sqrt(2) * x_prev1[4]
    result = np.array([x0, x1, x2, x3, x4]) + np.random.normal(loc=0.0, scale=np.sqrt(variance), size=5)
    return result

for variance in variances:
    results_mat = np.zeros((5, num_time_points+1))
    results_mat[:,0] = np.zeros(5)
    x_prev1 = np.zeros(5)
    x_prev2 = np.zeros(5)
    x_prev3 = np.zeros(5)
    x = np.zeros(5)
    for i in range(num_time_points):
        x_next = calc_next(x, x_prev1, x_prev2, x_prev3, variance)
        results_mat[:,i+1] = x_next
        x_prev3 = x_prev2
        x_prev2 = x_prev1
        x_prev1 = x
        x = x_next
    file_name = "variance_" + str(variance).replace('.','') + "_data.csv"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file_path = os.path.join(current_dir, file_name)
    np.savetxt(output_file_path, results_mat, delimiter=',')
    time_points = np.arange(0,11)
    plt.clf()
    for i in range(5):
        plt.plot(time_points, results_mat[i], label=f'x{i+1}')
    plot_file_name = "variance_" + str(variance).replace('.','') + "_plot"
    plt.savefig(plot_file_name)