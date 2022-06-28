from torch.autograd import Fuction

class GradientReversalLayer(Funtion):
    
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        output = - ctx.lam * grad_output
        
        return output, None
        