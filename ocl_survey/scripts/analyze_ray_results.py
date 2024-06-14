import os
import pandas as pd
from ray.tune import ExperimentAnalysis
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Directory where the ray.tune results are stored
experiment_dir = "/data/e.urettini/projects/CL_stability/ocl_survey/ray_results/train_function_2024-06-13_20-59-36"

# Load the results using ExperimentAnalysis
analysis = ExperimentAnalysis(experiment_dir)

# Get all the results in a DataFrame
df = analysis.dataframe()

#print column names
# Specify the parameter names
param1 = "config/optimizer/lr"
param2 = "config/strategy/regul"
metric_name = "final_accuracy"

# Extract relevant data
x = df[param1].values
y = df[param2].values
z = df[metric_name].values

# Transform both parameters to a logarithmic scale
x_log = np.log10(x)
y_log = np.log10(y)

# Create grid data for contour plot
xi = np.linspace(x_log.min(), x_log.max(), 100)
yi = np.linspace(y_log.min(), y_log.max(), 100)
zi = griddata((x_log, y_log), z, (xi[None, :], yi[:, None]), method='cubic')

# Find the best trial
best_config = analysis.get_best_config(metric=metric_name, mode="min")  # or mode="max" depending on your objective
best_x = best_config['optimizer']['lr']
best_y = best_config['strategy']['regul']
print(best_x, best_y)

# Create the contour plot with log scale for both axes
plt.figure(figsize=(10, 6))
contour = plt.contourf(10**xi, 10**yi, zi, levels=20, cmap='viridis')
plt.colorbar(contour)
plt.scatter(x, y, c=z, edgecolors='k', linewidths=0.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(param1)
plt.ylabel(param2)
plt.title(f'Contour plot of {metric_name}')

# Highlight the best configuration with a red X
plt.scatter(best_x, best_y, color='red', marker='x', s=100, label='Best Config')
plt.legend()
# Save the plot to a file
plot_filename = experiment_dir + "/contour_plot.png"
plt.savefig(plot_filename)
print(f"Contour plot saved to {plot_filename}")