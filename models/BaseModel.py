from __future__ import print_function
from more_itertools import first
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.nn import normal_init, NonLinear
from utils.distributions import log_normal_diag_vectorized
import math
from utils.nn import he_init
from utils.distributions import pairwise_distance
from utils.distributions import log_bernoulli, log_normal_diag, log_normal_standard, log_logistic_256
from utils.distributions_hypervae import *
from abc import ABC, abstractmethod
from torchvision import transforms


class BaseModel(nn.Module, ABC):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        print("constructor")
        self.args = args

        if self.args.prior == 'vampprior':
            self.add_pseudoinputs()

        if self.args.prior == 'exemplar_prior':
            self.prior_log_variance = torch.nn.Parameter(torch.randn((1)))
        
        if self.args.prior == 'trans_exemplar_prior':
            self.prior_log_variance = torch.nn.Parameter(torch.randn((1)))

        if self.args.prior == 'h_vae':
            self.prior_log_variance = torch.nn.Parameter(torch.randn((1)))

        if self.args.input_type == 'binary':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size))
            self.p_x_logvar = NonLinear(self.args.hidden_size, np.prod(self.args.input_size),
                                        activation=nn.Hardtanh(min_val=-4.5, max_val=0))
            self.decoder_logstd = torch.nn.Parameter(torch.tensor([0.], requires_grad=True))

        self.create_model(args)
        self.he_initializer()

    def he_initializer(self):
        print("he initializer")

        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

    @abstractmethod
    def create_model(self, args):
        pass

    @abstractmethod
    def kl_loss(self, latent_stats, exemplars_embeddin, dataset, cache, x_indices):
        pass
    
    def kl_loss_hyperVAE(self, latent_stats):
        _, z_q_mean, z_q_logvar = latent_stats
        q_z = VonMisesFisher(z_q_mean, z_q_logvar)
        p_z = HypersphericalUniform(self.args.z1_size - 1, device='cuda')
        KL = torch.distributions.kl.kl_divergence(q_z, p_z)
        return KL
    
    def kl_loss_trans_exemplar_hyper(self, latent_stats, exemplars_embedding, x=None):
        _, z_q_mean, z_q_logvar = latent_stats
        # z_q_var = torch.exp(z_q_logvar)
        q_z = VonMisesFisher(z_q_mean, z_q_logvar)
        first_part = -q_z.entropy()
        if exemplars_embedding == None:
            exemplars_embedding = self.get_trans_exemplar_set(x)
        
        set_mu, kappa = exemplars_embedding # set_mu:[M * D_z] kappa:scale z_q_mean:[B * D_z]
        # kappa = torch.exp(log_kappa)
        z_dim = z_q_mean.shape[-1]

        second_1 = kappa * ive(z_dim, z_q_logvar) / ive((z_dim / 2) - 1, z_q_logvar)

        second_2 = torch.mm(z_q_mean, set_mu.T)

        second_part = second_1 * second_2
        second_part = torch.log(torch.sum(torch.exp(second_part), dim=1))
        third_part = (z_dim / 2 - 1) * torch.log(kappa) - (z_dim / 2) * math.log(2 * math.pi) - (kappa + torch.log(ive(z_dim / 2 - 1, kappa)))
        forth_part = torch.log(torch.FloatTensor([len(set_mu)])).to(self.args.device)
        # print("first:", first_part.mean())
        # print("second:", second_part.mean())
        # print("third:", third_part)
        # print("forth:", forth_part)

        KL = first_part - second_part - third_part + forth_part
        return KL
        
    def reconstruction_loss(self, x, x_mean, x_logvar):
        if self.args.input_type == 'binary':
            return log_bernoulli(x, x_mean, dim=1)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            if self.args.use_logit is True:
                return log_normal_diag(x, x_mean, x_logvar, dim=1)
            else:
                return log_logistic_256(x, x_mean, x_logvar, dim=1)
        else:
            raise Exception('Wrong input type!')

    def calculate_loss(self, x, beta=1., average=False,
                       exemplars_embedding=None, cache=None, dataset=None):
        x, x_indices = x
        x_mean, x_logvar, latent_stats = self.forward(x)
        RE = self.reconstruction_loss(x, x_mean, x_logvar)
        if self.args.prior == 'h_vae':
            KL = self.kl_loss_trans_exemplar_hyper(latent_stats, exemplars_embedding, x)
            # KL = self.kl_loss_hyperVAE(latent_stats)
        else:
            KL = self.kl_loss(latent_stats, exemplars_embedding, dataset, cache, x_indices, x)
        loss = -RE + beta * KL
        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        return loss, RE, KL

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = mu.new_empty(size=std.shape).normal_()
        return eps.mul(std).add_(mu)

    def log_p_z_vampprior(self, z, exemplars_embedding):
        if exemplars_embedding is None:
            C = self.args.number_components
            X = self.means(self.idle_input)
            z_p_mean, z_p_logvar = self.q_z(X, prior=True)  # C x M
        else:
            C = torch.tensor(self.args.number_components).float()
            z_p_mean, z_p_logvar = exemplars_embedding

        z_expand = z.unsqueeze(1)
        means = z_p_mean.unsqueeze(0)
        logvars = z_p_logvar.unsqueeze(0)
        return log_normal_diag(z_expand, means, logvars, dim=2) - math.log(C)
    
    def log_p_z_trans_exemplar_vector(self, z, exemplars_embedding, test):
        centers, center_log_variance = exemplars_embedding
        center_log_variance = center_log_variance[0, :].unsqueeze(0)
        # print(z.shape) # [100, 40]
        # print(centers.shape) # [600, 40]
        # print(center_log_variance.shape) # [1, 40]
        # print(center_log_variance)
        prob, _ = log_normal_diag_vectorized(z, centers, center_log_variance)  # MB x C
        # print("prob", prob.shape) # [100, 600]
        if test:
            denominator = torch.tensor(len(centers)).expand(len(z)).float().to(self.args.device)
        else:
            denominator = torch.tensor(len(centers) / self.args.batch_size).expand(len(z)).float().to(self.args.device)
            # extract prob
            prob_list = []
            for i in range(prob.shape[0]):
                index = [i + k for k in range(0, prob.shape[1], self.args.batch_size)]
                prob_list.append(prob[i:i+1,index])
            prob = torch.cat(prob_list, dim=0)
        prob -= torch.log(denominator).unsqueeze(1)
        return prob

    def log_p_z_exemplar(self, z, z_indices, exemplars_embedding, test):
        centers, center_log_variance, center_indices = exemplars_embedding
        denominator = torch.tensor(len(centers)).expand(len(z)).float().to(self.args.device)
        center_log_variance = center_log_variance[0, :].unsqueeze(0)
        prob, _ = log_normal_diag_vectorized(z, centers, center_log_variance)  # MB x C
        if test is False and self.args.no_mask is False:
            mask = z_indices.expand(-1, len(center_indices)) \
                    == center_indices.squeeze().unsqueeze(0).expand(len(z_indices), -1)
            prob.masked_fill_(mask, value=float('-inf'))
            denominator = denominator - mask.sum(dim=1).float()
        prob -= torch.log(denominator).unsqueeze(1)
        return prob

    def log_p_z(self, z, exemplars_embedding, sum=True, test=None):
        z, z_indices = z
        if test is None:
            test = not self.training
        if self.args.prior == 'standard':
            return log_normal_standard(z, dim=1)
        elif self.args.prior == 'vampprior':
            prob = self.log_p_z_vampprior(z, exemplars_embedding)
        elif self.args.prior == 'exemplar_prior':
            prob = self.log_p_z_exemplar(z, z_indices, exemplars_embedding, test)
        elif self.args.prior == 'trans_exemplar_prior':
            prob = self.log_p_z_trans_exemplar_vector(z, exemplars_embedding, test)
            
        else:
            raise Exception('Wrong name of the prior!')
        if sum:
            prob_max, _ = torch.max(prob, 1)  # MB x 1
            log_prior = prob_max + torch.log(torch.sum(torch.exp(prob - prob_max.unsqueeze(1)), 1))  # MB x 1
        else:
            return prob
        return log_prior

    def add_pseudoinputs(self):
        nonlinearity = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.means = NonLinear(self.args.number_components, np.prod(self.args.input_size), bias=False, activation=nonlinearity)
        # init pseudo-inputs
        if self.args.use_training_data_init:
            self.means.linear.weight.data = self.args.pseudoinputs_mean
        else:
            normal_init(self.means.linear, self.args.pseudoinputs_mean, self.args.pseudoinputs_std)
        self.idle_input = Variable(torch.eye(self.args.number_components, self.args.number_components), requires_grad=False)
        self.idle_input = self.idle_input.to(self.args.device)

    def generate_z_interpolate(self, exemplars_embedding=None, dim=0):
        new_zs = []
        exemplars_embedding, _, _ = exemplars_embedding
        step_counts = 10
        step = (exemplars_embedding[1] - exemplars_embedding[0])/step_counts
        for i in range(step_counts):
            new_z = exemplars_embedding[0].clone()
            new_z += i*step
            new_zs.append(new_z.unsqueeze(0))
        return torch.cat(new_zs, dim=0)

    def generate_z(self, N=25, dataset=None):
        if self.args.prior == 'standard':
            z_sample_rand = torch.FloatTensor(N, self.args.z1_size).normal_().to(self.args.device)
        elif self.args.prior == 'vampprior':
            means = self.means(self.idle_input)[0:N]
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z(means)
            z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)
            z_sample_rand = z_sample_rand.to(self.args.device)
        elif self.args.prior == 'exemplar_prior':
            rand_indices = torch.randint(low=0, high=self.args.training_set_size, size=(N,))
            exemplars = dataset.tensors[0][rand_indices]
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z(exemplars.to(self.args.device), prior=True)
            z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)
            z_sample_rand = z_sample_rand.to(self.args.device)
        elif self.args.prior == 'trans_exemplar_prior':
            rand_indices = torch.randint(low=0, high=self.args.training_set_size, size=(N,))
            exemplars = dataset.tensors[0][rand_indices]
            z_sample_gen_mean, z_sample_gen_logvar = self.q_z(exemplars.to(self.args.device), prior=True)
            z_sample_rand = self.reparameterize(z_sample_gen_mean, z_sample_gen_logvar)
            z_sample_rand = z_sample_rand.to(self.args.device)
        elif self.args.prior == 'h_vae':
            output = (
                torch.distributions.Normal(0, 1)
                .sample(
                    torch.Size([N]) + torch.Size([self.args.z1_size])
                )
                .to(self.args.device)
            )
            z_sample_rand = output / output.norm(dim=-1, keepdim=True)
        
        return z_sample_rand

    def reference_based_generation_z(self, N=25, reference_image=None):
        pseudo, log_var = self.q_z(reference_image.to(self.args.device), prior=True)
        pseudo = pseudo.unsqueeze(1).expand(-1, N, -1).reshape(-1, pseudo.shape[-1])
        log_var = log_var[0].unsqueeze(0).expand(len(pseudo), -1)
        z_sample_rand = self.reparameterize(pseudo, log_var)
        z_sample_rand = z_sample_rand.reshape(-1, N, pseudo.shape[1])
        return z_sample_rand

    def reconstruct_x(self, x):
        x_reconstructed, _, z = self.forward(x)
        if self.args.model_name == 'pixelcnn':
            x_reconstructed = self.pixelcnn_generate(z[0].reshape(-1, self.args.z1_size), z[3].reshape(-1, self.args.z2_size))
        return x_reconstructed

    def logit_inverse(self, x):
        sigmoid = torch.nn.Sigmoid()
        lambd = self.args.lambd
        return ((sigmoid(x) - lambd)/(1-2*lambd))

    def generate_x(self, N=25, dataset=None):
        z2_sample_rand = self.generate_z(N=N, dataset=dataset)
        return self.generate_x_from_z(z2_sample_rand)

    def reference_based_generation_x(self, N=25, reference_image=None):
        z2_sample_rand = \
            self.reference_based_generation_z(N=N, reference_image=reference_image)
        generated_x = self.generate_x_from_z(z2_sample_rand)
        return generated_x

    def generate_x_interpolate(self, exemplars_embedding, dim=0):
        zs = self.generate_z_interpolate(exemplars_embedding, dim=dim)
        print(zs.shape)
        return self.generate_x_from_z(zs, with_reparameterize=False)

    def reshape_variance(self, variance, shape):
        return variance[0]*torch.ones(shape).to(self.args.device)

    def q_z(self, x, prior=False):
        if  'conv' in self.args.model_name or 'pixelcnn'==self.args.model_name:
            x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        h = self.q_z_layers(x)
        if self.args.model_name == 'convhvae_2level' or self.args.model_name=='pixelcnn':
            h = h.view(x.size(0), -1)
        z_q_mean = self.q_z_mean(h)
        if prior is True:
            if self.args.prior == 'exemplar_prior':
                z_q_logvar = self.prior_log_variance * torch.ones((x.shape[0], self.args.z1_size)).to(self.args.device)
                if self.args.model_name == 'newconvhvae_2level':
                    z_q_logvar = z_q_logvar.reshape(-1, 4, 4, 4)
            elif self.args.prior == 'trans_exemplar_prior':
                z_q_logvar = self.prior_log_variance * torch.ones((x.shape[0], self.args.z1_size)).to(self.args.device)
            elif self.args.prior == 'h_vae':
                z_q_logvar = self.prior_log_variance
            else:
                z_q_logvar = self.q_z_logvar(h)
        else:
            z_q_logvar = self.q_z_logvar(h)
        if self.args.prior == 'h_vae':
            return z_q_mean, z_q_logvar
        else:
            return z_q_mean.reshape(-1, self.args.z1_size), z_q_logvar.reshape(-1, self.args.z1_size)

    def cache_z(self, dataset, prior=True, cuda=True):
        cached_z = []
        cached_log_var = []
        caching_batch_size = 10000
        num_batchs = math.ceil(len(dataset) / caching_batch_size)
        for i in range(num_batchs):
            if len(dataset[0]) == 3:
                batch_data, batch_indices, _ = dataset[i * caching_batch_size:(i + 1) * caching_batch_size]
            else:
                batch_data, _ = dataset[i * caching_batch_size:(i + 1) * caching_batch_size]

            exemplars_embedding, log_variance_z = self.q_z(batch_data.to(self.args.device), prior=prior)
            cached_z.append(exemplars_embedding)
            if self.args.prior != 'h_vae':
                cached_log_var.append(log_variance_z)
            elif i == 0:
                cached_log_var.append(log_variance_z)
        cached_z = torch.cat(cached_z, dim=0)
        cached_log_var = torch.cat(cached_log_var, dim=0)
        cached_z = cached_z.to(self.args.device)
        cached_log_var = cached_log_var.to(self.args.device)
        return cached_z, cached_log_var

    def get_exemplar_set(self, z_mean, z_log_var, dataset, cache, x_indices):
        if self.args.approximate_prior is False:
            exemplars_indices = torch.randint(low=0, high=self.args.training_set_size,
                                              size=(self.args.number_components, ))
            exemplars_z, log_variance = self.q_z(dataset.tensors[0][exemplars_indices].to(self.args.device), prior=True)
            exemplar_set = (exemplars_z, log_variance, exemplars_indices.to(self.args.device))
        else:
            exemplar_set = self.get_approximate_nearest_exemplars(
                z=(z_mean, z_log_var, x_indices),
                dataset=dataset,
                cache=cache)
        return exemplar_set
    
    def images_transfroms(self, input_images, num_iter=1, with_input = True):


        transform = transforms.Compose([
                # transforms.RandomResizedCrop(size=28),
                transforms.RandomResizedCrop(size=28, scale=(0.08, 0.2)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=25),
                transforms.GaussianBlur(kernel_size=3),
                # transforms.RandomGrayscale(p=0.2),
                ])
        
        
        for i in range(num_iter):
            output_image = transform(input_images)
            if i == 0:
                if with_input:
                    output_images = torch.cat((input_images, output_image), dim=0)
                else:
                    output_images = output_image
            else:
                output_images = torch.cat((output_images, output_image), dim=0)
        return output_images
    
    def get_trans_exemplar_set(self, images):
        images = images.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        trans_images = self.images_transfroms(images, num_iter=10, with_input=False)
        trans_images = trans_images.view(-1, np.prod(self.args.input_size))
        # x to z
        exemplars_z, log_variance = self.q_z(trans_images.float().to(self.args.device), prior=True)
        if self.args.prior == 'h_vae':
            exemplars_z = exemplars_z / exemplars_z.norm(dim=-1, keepdim=True)
            log_variance = F.softplus(log_variance) + 1
            
        
        exemplar_set = (exemplars_z, log_variance)

        return exemplar_set

    def get_approximate_nearest_exemplars(self, z, cache, dataset):
        exemplars_indices = torch.randint(low=0, high=self.args.training_set_size,
                                          size=(self.args.number_components, )).to(self.args.device)
        z, _, indices = z
        cached_z, cached_log_variance = cache
        cached_z[indices.reshape(-1)] = z
        sub_cache = cached_z[exemplars_indices, :]
        _, nearest_indices = pairwise_distance(z, sub_cache) \
            .topk(k=self.args.approximate_k, largest=False, dim=1)
        nearest_indices = torch.unique(nearest_indices.view(-1))
        exemplars_indices = exemplars_indices[nearest_indices].view(-1)
        exemplars = dataset.tensors[0][exemplars_indices].to(self.args.device)
        exemplars_z, log_variance = self.q_z(exemplars, prior=True)
        cached_z[exemplars_indices] = exemplars_z
        exemplar_set = (exemplars_z, log_variance, exemplars_indices)
        return exemplar_set
