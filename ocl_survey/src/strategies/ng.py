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

    def __init__(self, representation, regul, variant='classif_logits'):
        super().__init__()
        self.representation = eval(representation)
        self.variant = variant
        self.regul = regul

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

        original_grad_vec = PVector.from_model_grad(strategy.model)
        regularized_grad = F.solve(original_grad_vec, regul=self.regul) 
        regularized_grad.to_model_grad(strategy.model)  


    