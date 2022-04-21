from __future__ import print_function
from readline import parse_and_bind
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from models.BaseModel import BaseModel
from utils.distributions import log_normal_diag
from utils.distributions_hypervae import *

class AbsModel(BaseModel):
    def __init__(self, args):
        super(AbsModel, self).__init__(args)

    def kl_loss(self, latent_stats, exemplars_embedding, dataset, cache, x_indices, x=None):
        z_q, z_q_mean, z_q_logvar = latent_stats
        if exemplars_embedding is None and self.args.prior == 'exemplar_prior':
            exemplars_embedding = self.get_exemplar_set(z_q_mean, z_q_logvar, dataset, cache, x_indices)
        elif exemplars_embedding is None and self.args.prior == 'trans_exemplar_prior':
            exemplars_embedding = self.get_trans_exemplar_set(x)
        log_p_z = self.log_p_z(z=(z_q, x_indices), exemplars_embedding=exemplars_embedding)
        log_q_z = log_normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
        return -(log_p_z - log_q_z)

    def generate_x_from_z(self, z, with_reparameterize=True):
        generated_x, _ = self.p_x(z)
        try:
            if self.args.use_logit is True:
                return self.logit_inverse(generated_x)
            else:
                return generated_x
        except:
            return generated_x

    def p_x(self, z):
        if 'conv' in self.args.model_name:
            z = z.reshape(-1, self.bottleneck, self.args.input_size[1]//4, self.args.input_size[1]//4)
        z = self.p_x_layers(z)
        x_mean = self.p_x_mean(z)
        if self.args.input_type == 'binary':
            x_logvar = torch.zeros(1, np.prod(self.args.input_size))
        else:
            if self.args.use_logit is False:
                x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
                x_logvar = self.decoder_logstd*x_mean.new_ones(size=x_mean.shape)
        return x_mean.reshape(-1, np.prod(self.args.input_size)), x_logvar.reshape(-1, np.prod(self.args.input_size))

    def forward(self, x, label=0, num_categories=10):
        z_q_mean, z_q_logvar = self.q_z(x)
        if self.args.prior == 'h_vae':
            # Scale to hypersphere space: L2 norm
            z_q_mean = z_q_mean / z_q_mean.norm(dim=-1, keepdim=True)
            z_q_logvar = F.softplus(z_q_logvar) + 1
            # z_q_var = torch.exp(z_q_logvar)
            q_z = VonMisesFisher(z_q_mean, z_q_logvar)
            z_q = q_z.rsample()
        else:
            z_q = self.reparameterize(z_q_mean, z_q_logvar)
        x_mean, x_logvar = self.p_x(z_q)
        return x_mean, x_logvar, (z_q, z_q_mean, z_q_logvar)
