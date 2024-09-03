from avalanche.training.plugins import SupervisedPlugin
import torch
import csv


class SignSGDPlugin(SupervisedPlugin):
    def __init__(self):
        super().__init__()

        # If file csv file exists, delete it
        import os
        file_path = '../../tau_grads.csv'
        if os.path.exists(file_path):
            os.remove(file_path)
        # Create a new csv file with the header
        with open('../../tau_grads.csv', mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['original_last','regul_last'])



    def before_update(self, strategy, **kwargs):
        """
        This method will be called before each update step to modify the gradients.
        Args:
            strategy: The strategy object containing the optimizer and other components.
            kwargs: Additional arguments.
        """

        original_last = torch.norm(strategy.model.linear.classifier.weight.grad.flatten())

        
        
        torch.nn.utils.clip_grad_norm_(strategy.model.parameters(), 0.15)


        regul_last = torch.norm(strategy.model.linear.classifier.weight.grad.flatten())
        with open('../../tau_grads.csv', mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([original_last.item(), regul_last.item()])


