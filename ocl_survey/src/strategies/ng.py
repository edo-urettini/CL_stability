from avalanche.training.plugins import SupervisedPlugin
import torch
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag, PMatBlockDiag, PMatKFAC, PMatEKFAC, PMatDense, PMatQuasiDiag, PVector



class NGPlugin(SupervisedPlugin):
    """
    SupervisedPlugin for Fisher Information Regularization.

    This plugin modifies gradients before optimizer update by
    multiplying them with the inverse of the Fisher Information matrix (FIM).

    Args:
        model (torch.nn.Module): The model to be trained.
        representation (str, optional): The model representation to use for 
            FIM calculation (e.g., 'PMatKFAC'). Defaults to PMatKFAC.
        n_output (int, optional): The number of output classes. Defaults to None.
        variant (str, optional): The variant of the FIM calculation (e.g., 'classif_logits').
            Defaults to 'classif_logits'.
    """

    def __init__(self, representation, regul, alpha_ema, variant='classif_logits'):
        super().__init__()
        self.representation = eval(representation)
        self.variant = variant
        self.regul = regul
        self.alpha_ema = alpha_ema
        self.F_ema = None


    def EMA_kfac(self, mat_ema, mat_new, alpha):
        """
        Compute the exponential moving average of two PMatKFAC matrices.

        :param pmat_kfac1: The first PMatKFAC matrix.
        :param pmat_kfac2: The second PMatKFAC matrix.
        :param alpha: The smoothing factor for the EMA.
        :return: A new PMatKFAC matrix representing the EMA.
        """
        shared_keys = mat_ema.data.keys() & mat_new.data.keys()

        ema_data = {}
        for layer_id in shared_keys:
            a1, g1 = mat_ema.data[layer_id]
            a2, g2 = mat_new.data[layer_id]

            ema_a = alpha * a1 + (1 - alpha) * a2
            ema_g = alpha * g1 + (1 - alpha) * g2

            mat_new.data[layer_id] = (ema_a, ema_g)

        # Create a new PMatKFAC instance with the EMA data
        return mat_new

# Example usage:
# pmat_kfac_ema = exponential_moving_average(pmat_kfac1, pmat_kfac2, alpha=0.3)


    def before_update(self, strategy, **kwargs):
        """
        Modifies gradients before optimizer update.

        Args:
            strategy (Template): The current training strategy.
        """
        if not strategy.train:
            return        
        

        #Create a temporary dataloader to calculate the FIM
        temp_dataloader = torch.utils.data.TensorDataset(strategy.mb_x, strategy.mb_y)
        temp_dataloader = torch.utils.data.DataLoader(temp_dataloader, batch_size=strategy.mb_x.size(0), shuffle=False)


        F = FIM(model=strategy.model,
                loader=temp_dataloader,
                representation=self.representation,
                n_output=strategy.mb_output.size(1),
                variant=self.variant, 
                device=strategy.device)
        
        #Update the EMA of the FIM
        if self.F_ema is None or self.alpha_ema == 0.0:
            self.F_ema = F
        else:
            self.F_ema = self.EMA_kfac(self.F_ema, F, self.alpha_ema)

        original_grad_vec = PVector.from_model_grad(strategy.model)
        regularized_grad = self.F_ema.solve(original_grad_vec, regul=self.regul) 
        regularized_grad.to_model_grad(strategy.model)  


    