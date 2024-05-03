import torch
import torch.nn as nn
from functorch import jvp, make_functional_with_buffers

class LinearizedModel(nn.Module):
    def __init__(self, model, init_model: nn.Module):
        super(LinearizedModel, self).__init__()
        self.model = model
        self.init_model = init_model
        # init_model = init_model or model
        if self.init_model is None:
            self.init_model = self.model

        func0, params0, self.buffers0 = make_functional_with_buffers(self.init_model.eval(), disable_autograd_tracking=True)
        self.func0 = lambda params, x: func0(params, self.buffers0, x)

        _, params, _ = make_functional_with_buffers(self.model, disable_autograd_tracking=True)


        self.params =  nn.ParameterList(params)
        self.params0 = nn.ParameterList(params0)

        # The intial parameters are not trainable.
        for p in self.params0:
            p.requires_grad = False

        # The params are.
        for p in self.params:
            p.requires_grad = True

    def __call__(self, x) -> torch.Tensor:

        # print("Computes the linearized model output using a first-order Taylor decomposition.")
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        out, dp = jvp(lambda param: self.func0(param, x),(tuple(self.params0),),(tuple(dparams),),)

        return out + dp

    def disable_bn_tracking(self):
        def _disable(m):
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
        self.apply(_disable)

