import torch

from .solver import DDIMSolver, HybridSolver,FMSolver
import numpy as np


def unsqueeze_xdim(z, xdim):
    """
    Add singleton dimensions to the tensor `z` to match the length of `xdim`.

    Args:
        z (torch.Tensor): The input tensor.
        xdim (tuple): The target dimensions to be unsqueezed.

    Returns:
        torch.Tensor: The unsqueezed tensor.
    """
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


class DiffusionBridgeSDE():
    def __init__(self, schedule_type='VE', c=0.4, k=2.6, beta0=0.01, beta1=20, t_min=1e-4, t_max=1, loss_weight_type=None, device='cpu', method='SB'):
        """Construct a Variance Preserving SDE.
        # 可选参数包括VE、VP和gmax
        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
        """
        self.method = method #可选模型 SB、FM
        self.device = device
        if schedule_type=='VE':
            self.c = 0.4
        else:
            self.c = 0.3

        self.k = k
        self.beta0 = beta0
        self.beta1 = beta1
        self.schedule_type = schedule_type

        self.t_min = t_min
        self.t_max = t_max
        if loss_weight_type is None:
            loss_weight_type = 'constant'
        self.loss_weight_type = loss_weight_type
        self.eps = 1e-8

        # flow matching
        self.sigma_min = 0
        self.sigma_max = 0.2

    def alpha_t(self, t):
        if self.schedule_type == 'VP':
            return torch.exp(-0.5 * (self.beta0 * t + (self.beta1 - self.beta0) * t**2 / 2))
        elif self.schedule_type == 'VE':
            return torch.ones_like(t)
        elif self.schedule_type == 'gmax':
            return torch.ones_like(t)

    def sigma_t(self, t):
        if self.schedule_type == 'VP':
            return torch.sqrt(self.c * (torch.exp(self.beta0 * t + (self.beta1 - self.beta0) * t**2 / 2) - 1))
        elif self.schedule_type == 'VE':
            return torch.sqrt(self.c * (self.k**(2*t) - 1) / (2 * np.log(self.k)))
        elif self.schedule_type == 'gmax':
            return torch.sqrt((t**2 * (self.beta1 - self.beta0)) / 2 + self.beta0 * t)

    # def g2_t(self, t):
    #     if self.schedule_type == 'VP':
    #         return self.c * (self.beta0 + (self.beta1 - self.beta0) * t)
    #     elif self.schedule_type == 'VE':
    #         return self.c * self.k**(2*t)
    #     elif self.schedule_type == 'gmax':
    #         return self.beta0 + t * (self.beta1 - self.beta0)

    def q_sample(self, t, x0, x1, ot_ode=False, x0_bar=None):
        if self.method=="SB":
            batch, *xdim = x0.shape

            alpha_t = self.alpha_t(t)
            sigma_t = self.sigma_t(t)
            sigma_T = self.sigma_t(torch.ones_like(t) * 1.0)  # T=1
            alpha_T = self.alpha_t(torch.ones_like(t) * 1.0)  # T=1

            w_x = alpha_t * (sigma_T**2 - sigma_t**2 + self.eps) / (sigma_T**2 + self.eps)
            w_y = (alpha_t / (alpha_T + self.eps)) * sigma_t**2 / (sigma_T**2 + self.eps)
            var = (alpha_t**2 * (sigma_T**2 - sigma_t**2 + self.eps) * sigma_t**2) / (sigma_T**2 + self.eps)

            w_x = unsqueeze_xdim(w_x, xdim)
            w_y = unsqueeze_xdim(w_y, xdim)
            var = unsqueeze_xdim(var, xdim)

            x_t = w_x * x0 + w_y * x1

            if not ot_ode:
                x_t += var.sqrt() * torch.randn_like(x_t)

            return x_t
        
        elif self.method == "FM":
            batch, *xdim = x0.shape
            mean, std = (1-t)[:,None,None,None]*x0 + t[:,None,None,None]*x1
            z = torch.randn_like(x0)  #
            sigmas = std[:, None, None, None]
            x_t = mean + sigmas * z

            return x_t

    def p_posterior(self, t, s, x, x0, ot_ode=False, x1=None, delta_t=None, randn=True):
        # x0 = self.marginal_alpha(t) / self.marginal_alpha(s) * x0
        if self.method=="SB":
            if not ot_ode:
                print('现在你正在使用sb-sde解码器！')
                alpha_t = self.alpha_t(t)
                sigma_t = self.sigma_t(t)
                sigma_s = self.sigma_t(s)
                alpha_s = self.alpha_t(s)

                # SDE更新步骤
                w_x = (alpha_t * sigma_t**2) / (alpha_s * sigma_s**2 + self.eps)
                w_y = alpha_t * (1 - sigma_t**2 / (sigma_s**2 + self.eps))

                batch, *xdim = x0.shape

                w_x = unsqueeze_xdim(w_x, xdim)
                w_y = unsqueeze_xdim(w_y, xdim)

                term1 = w_x * x
                term2 = w_y * x0
                noise = torch.randn_like(x)
                term3 = alpha_t  * torch.sqrt(sigma_t**2 * (1 - sigma_t**2 / (sigma_s**2 + self.eps))) * noise

                if randn:
                    x_prev = term1 + term2 + term3
                else:
                    x_prev = term1 + term2# + term3


                return x_prev
            else:
                # 计算所有时间相关参数
                alpha_t = self.alpha_t(t)
                sigma_t = self.sigma_t(t)

                alpha_tau = self.alpha_t(s)
                sigma_tau = self.sigma_t(s)

                alpha_T = self.alpha_t(torch.tensor(1.0, device=x.device))
                sigma_T = self.sigma_t(torch.tensor(1.0, device=x.device))

                bar_sigma_t = torch.sqrt(sigma_T**2 - sigma_t**2 + self.eps)
                bar_sigma_tau = torch.sqrt(sigma_T**2 - sigma_tau**2 + self.eps)

                # 严格按照表2的ODE更新公式
                coeff1 = (alpha_t * sigma_t * bar_sigma_t) / \
                        (alpha_tau * sigma_tau * bar_sigma_tau + self.eps)
                coeff2 = (alpha_t / (sigma_T**2 + self.eps)) * \
                        (bar_sigma_t**2 - (bar_sigma_tau * sigma_t * bar_sigma_t) / (sigma_tau + self.eps))
                coeff3 = (alpha_t / (alpha_T * sigma_T**2 + self.eps)) * \
                        (sigma_t**2 - (sigma_tau * sigma_t * bar_sigma_t) / (bar_sigma_tau + self.eps))

                batch, *xdim = x0.shape

                coeff1 = unsqueeze_xdim(coeff1, xdim)
                coeff2 = unsqueeze_xdim(coeff2, xdim)
                coeff3 = unsqueeze_xdim(coeff3, xdim)

                term1 =  coeff1 * x
                term2 =  coeff2* x0
                term3 = coeff3 * x1
                x_prev = term1 + term2 + term3
                return x_prev
            
        elif self.method=="FM":
            # pred 是x0，当前是x
            batch, *xdim = x0.shape
            # t = unsqueeze_xdim(t, xdim)
            print(s)
            u_t = (x0 - x) / (1-s) #(1 - (1 - self.sigma_min) * t)
            # print(t,t,s)
            return x + u_t*delta_t
            # return x0

    def compute_pred_x0(self, t, xt, net_out, clip_denoise=False):
        alpha_t, sigma_t = self.marginal_alpha(t), self.marginal_sigma(t)

        batch, *xdim = xt.shape
        alpha_t = unsqueeze_xdim(alpha_t, xdim)
        sigma_t = unsqueeze_xdim(sigma_t, xdim)

        pred_x0 = (xt - sigma_t * net_out) / alpha_t
        return pred_x0

    def compute_label(self, t, x0, xt, x0_hat=None):
        xt = xt.detach()
        alpha_t, sigma_t = self.marginal_alpha(t), self.marginal_sigma(t)

        batch, *xdim = x0.shape
        alpha_t = unsqueeze_xdim(alpha_t, xdim)
        sigma_t = unsqueeze_xdim(sigma_t, xdim)
        if x0_hat is not None:
            x0_hat = x0_hat.detach()
            label = (xt - x0_hat * alpha_t) / sigma_t
        else:
            label = (xt - x0 * alpha_t) / sigma_t

        return label

    def compute_weight(self, t):
        if self.loss_weight_type == 'constant':
            mse_loss_weight = torch.ones_like(t, device=self.device)
        elif self.loss_weight_type == 'snr':
            mse_loss_weight = torch.exp(self.marginal_logSNR())
        elif self.loss_weight_type.startswith("min_snr_"):
            k = float(self.loss_weight_type.split('min_snr_')[-1])
            snr = torch.exp(self.marginal_logSNR(t))
            mse_loss_weight = torch.stack([snr, k * torch.ones_like(t)], dim=1).min(dim=1)[0] / snr
        else:
            raise NotImplementedError(f'unsupported weight type {self.loss_weight_type}')
        return mse_loss_weight

    def get_ddim_solver(self, model_fn, num_step=50, ot_ode=False):
        return DDIMSolver(self, model_fn=model_fn, num_step=num_step, ot_ode=ot_ode)

    def get_fm_solver(self, model_fn, num_step=50, ot_ode=False):
        return FMSolver(self, model_fn=model_fn, num_step=num_step, ot_ode=ot_ode)



class VP_DiffusionBridgeSDE(DiffusionBridgeSDE):
    def __init__(self, beta=0.1, t_min=3e-2, t_max=1, loss_weight_type=None, device='cpu'):
        super().__init__(beta=beta, t_min=t_min, t_max=t_max, loss_weight_type=loss_weight_type, device=device)

    def marginal_log_alpha(self, t):
        return - 0.5 * t * self.beta

    def marginal_log_sigma(self, t):
        return 0.5 * torch.log(1. - torch.exp(2. * self.marginal_log_alpha(t)))

    def get_hybrid_solver(self, model_fn=None, num_step=50, skip_type='time_uniform', ot_ode=False):
        return HybridSolver(sde=self, model_fn=model_fn, num_step=num_step, skip_type=skip_type, ot_ode=ot_ode, device=self.device)


class VE_DiffusionBridgeSDE(DiffusionBridgeSDE):
    def __init__(self, beta=0.1, t_max=1, loss_weight_type=None, device='cpu'):
        super().__init__(beta=beta, t_max=t_max, loss_weight_type=loss_weight_type, device=device)

    def marginal_log_alpha(self, t):
        return torch.zeros_like(t, device=self.device)

    def marginal_log_sigma(self, t):
        return torch.log(t)
