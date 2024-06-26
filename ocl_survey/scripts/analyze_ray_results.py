import os
import pandas as pd
from ray.tune import ExperimentAnalysis
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main(args):

    tune_result_dir = args.ray_results_dir
    sigma = args.sigma
    param1 = args.param1
    param2 = args.param2
    metric_name = "final_accuracy"

    log_list = ["config/optimizer/lr", "config/strategy/regul"]

    # Load the results using ExperimentAnalysis
    analysis = ExperimentAnalysis(tune_result_dir)

    # Get all the results in a DataFrame
    df = analysis.dataframe()

    # Extract relevant data
    x = df[param1].values
    z = df[metric_name].values

    # Check if param1 and param2 are in log scale
    is_param1_log = param1 in log_list
    x_log = np.log10(x) if is_param1_log else x

    # Check if param2 exists in the DataFrame
    if param2 is not None:
        y = df[param2].values
        is_param2_log = param2 in log_list
        y_log = np.log10(y) if is_param2_log else y
        
        # Create grid data for contour plot
        xi = np.linspace(x_log.min(), x_log.max(), 100)
        yi = np.linspace(y_log.min(), y_log.max(), 100)
        xi, yi = np.meshgrid(xi, yi)
        
        # Define a Gaussian kernel function
        def gaussian_kernel(d, sigma=sigma):
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
        contour = plt.contourf(10**xi if is_param1_log else xi, 10**yi if is_param2_log else yi, zi, levels=20, cmap='plasma')
        plt.colorbar(contour)
        plt.scatter(x, y, c=z, edgecolors='k', linewidths=0.5, cmap='plasma')
        if is_param1_log:
            plt.xscale('log')
        if is_param2_log:
            plt.yscale('log')
        plt.xlabel(param1)
        plt.ylabel(param2)
        
        # Find the best trial and highlight it on the plot
        best_config = analysis.get_best_config(metric=metric_name, mode="max")
        param1_folder = param1.split("/")[1]
        param2_folder = param2.split("/")[1]
        param1_key = param1.split("/")[2]
        param2_key = param2.split("/")[2]
        best_x = best_config[param1_folder][param1_key]
        best_y = best_config[param2_folder][param2_key]
        plt.scatter(best_x, best_y, color='red', marker='x', s=100, label='Best Config')

        
    else:
        # Create grid data for line plot
        xi = np.linspace(x_log.min(), x_log.max(), 100)
        
        # Define a Gaussian kernel function
        def gaussian_kernel(d, sigma=sigma):
            return np.exp(-0.5 * (d / sigma) ** 2)
        
        # Compute the weighted average at each grid point
        zi = np.zeros_like(xi)
        for i in range(xi.shape[0]):
            distances = np.abs(xi[i] - x_log)
            weights = gaussian_kernel(distances)
            zi[i] = np.sum(weights * z) / np.sum(weights)
        
        # Create the line plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(x, z, edgecolors='k', linewidths=0.5, label='Data', cmap='plasma')
        plt.plot(10**xi if is_param1_log else xi, zi, label='Weighted Average', color='red')
        if is_param1_log:
            plt.xscale('log')
        plt.xlabel(param1)
        plt.ylabel(metric_name)

        
        # Find the best trial and highlight it on the plot
        best_config = analysis.get_best_config(metric=metric_name, mode="max")
        param1_folder = param1.split("/")[1]
        param1_key = param1.split("/")[2]
        best_x = best_config[param1_folder][param1_key]
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
    parser.add_argument("param1", type=str)
    parser.add_argument("--param2", type=str, default=None)
    parser.add_argument("--sigma", type=float, default=0.3)
    args = parser.parse_args()
    main(args)