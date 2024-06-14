import os
import pandas as pd
from ray.tune import ExperimentAnalysis
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main(args):

    tune_result_dir = args.ray_results_dir
    # Load the results using ExperimentAnalysis
    analysis = ExperimentAnalysis(tune_result_dir)

    # Get all the results in a DataFrame
    df = analysis.dataframe()

    # Specify the parameter names
    param1 = "config/optimizer/lr"
    param2 = "config/strategy/regul"
    metric_name = "final_accuracy"

    # Extract relevant data
    x = df[param1].values
    z = df[metric_name].values

    # Transform parameter to a logarithmic scale
    x_log = np.log10(x)

    # Check if param2 exists in the DataFrame
    if param2 in df.columns:
        y = df[param2].values
        y_log = np.log10(y)
        
        # Create grid data for contour plot
        xi = np.linspace(x_log.min(), x_log.max(), 100)
        yi = np.linspace(y_log.min(), y_log.max(), 100)
        xi, yi = np.meshgrid(xi, yi)
        
        # Define a Gaussian kernel function
        def gaussian_kernel(d, sigma=0.3):
            return np.exp(-0.5 * (d / sigma) ** 2)
        
        # Compute the weighted average at each grid point
        zi = np.zeros_like(xi)
        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                distances = np.sqrt((xi[i, j] - x_log) ** 2 + (yi[i, j] - y_log) ** 2)
                weights = gaussian_kernel(distances)
                zi[i, j] = np.sum(weights * z) / np.sum(weights)
        
        # Create the contour plot with log scale for both axes
        plt.figure(figsize=(10, 6))
        contour = plt.contourf(10**xi, 10**yi, zi, levels=20, cmap='plasma')
        plt.colorbar(contour)
        plt.scatter(x, y, c=z, edgecolors='k', linewidths=0.5, cmap='plasma')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(param1)
        plt.ylabel(param2)
        
        # Find the best trial and highlight it on the plot
        best_config = analysis.get_best_config(metric=metric_name, mode="max")
        best_x = best_config['optimizer']['lr']
        best_y = best_config['strategy']['regul']
        plt.scatter(best_x, best_y, color='red', marker='x', s=100, label='Best Config')
    else:
        # Create grid data for line plot
        xi = np.linspace(x_log.min(), x_log.max(), 100)
        
        # Define a Gaussian kernel function
        def gaussian_kernel(d, sigma=0.3):
            return np.exp(-0.5 * (d / sigma) ** 2)
        
        # Compute the weighted average at each grid point
        zi = np.zeros_like(xi)
        for i in range(xi.shape[0]):
            distances = np.abs(xi[i] - x_log)
            weights = gaussian_kernel(distances)
            zi[i] = np.sum(weights * z) / np.sum(weights)
        
        # Create the line plot with log scale and color scale for accuracy
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(x, z, edgecolors='k', linewidths=0.5, label='Data', cmap='plasma')
        plt.plot(10**xi, zi, label='Interpolated')
        plt.colorbar(scatter)
        plt.xscale('log')
        plt.xlabel(param1)
        plt.ylabel(metric_name)
        
        # Find the best trial and highlight it on the plot
        best_config = analysis.get_best_config(metric=metric_name, mode="max")
        best_x = best_config['optimizer']['lr']
        best_y = np.max(z)  # Since there's no regul, use the max z value
        plt.scatter(best_x, best_y, color='red', marker='x', s=100, label='Best Config')

    # Finalize the plot
    plt.title(f'Plot of {metric_name}')
    plt.legend()

    # Save the plot to a file
    plot_filename = tune_result_dir + "/plot.png"
    plt.savefig(plot_filename)
    print(f"Contour plot saved to {plot_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ray_results_dir", type=str)
    args = parser.parse_args()
    main(args)