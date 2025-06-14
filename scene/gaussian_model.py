import torch
import numpy as np
from torch import nn
import os
from plyfile import PlyData, PlyElement


from utils.graphics_utils import BasicPointCloud
from simple_knn._C import distCUDA2
from utils.general_utils import strip_symmetric, build_scaling_rotation, \
    inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.system_utils import mkdir_p

from .pos_encoder import Update_SH_Coeffs

from utils.train_utils import initialize_weights
from utils.sh_utils import RGB2SH


class GaussianModel:


    def setup_functions(self):
        
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):

            L = build_scaling_rotation(scaling_modifier * scaling, rotation)  # torch.Size([N, 3, 3])

            actual_covariance = L @ L.transpose(1, 2)  # torch.Size([N, 3, 3])

            symm = strip_symmetric(actual_covariance)

            return symm, actual_covariance
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        # self.opacity_activation = torch.sigmoid
        # self.inverse_opacity_activation = inverse_sigmoid
        self.attenuation_activation = torch.sigmoid
        self.inverse_attenuation_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, args):

        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree
        
        self.data_device = args.data_device

        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)

        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)

        self._attenuation = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.setup_functions()

        self.input_dim_emd  = args.input_dim_emd        
        self.hidden_dim_1 = args.hidden_dim_1   
        self.hidden_dim_2 = args.hidden_dim_2   
        self.output_dim = args.output_dim

        self.total_params = (self.input_dim_emd * self.hidden_dim_1) + self.hidden_dim_1 \
            + (self.hidden_dim_1 * self.hidden_dim_2) + self.hidden_dim_2 \
                + (self.hidden_dim_2 * self.output_dim) + self.output_dim

        self.num_channels = 2

    @property
    def get_xyz(self):
        return self._xyz
    

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)


    @property
    def get_attenuation(self):

        return self.attenuation_activation(self._attenuation)

    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)


    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    

    def load_from_pcd(self, 
                        pcd : BasicPointCloud, 
                        spatial_lr_scale : float): 

        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()    # torch.Size([N, 3])

        features = torch.randn((fused_point_cloud.shape[0], self.num_channels, (self.max_sh_degree + 1) ** 2)).float().cuda()

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)

        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        attenuations = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz           = nn.Parameter(fused_point_cloud.requires_grad_(True))

        self._features_dc   = nn.Parameter(features[:, :, 0: 1].transpose(1, 2).contiguous().requires_grad_(True))  # torch.Size([182686, 3, 1])  => torch.Size([182686, 1, 3])
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))    # torch.Size([182686, 3, 15]) => torch.Size([182686, 15, 3])
        
        self._attenuation   = nn.Parameter(attenuations.requires_grad_(True))

        self._scaling       = nn.Parameter(scales.requires_grad_(True))

        self._rotation      = nn.Parameter(rots.requires_grad_(True))

        self.max_radii2D    = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def load_from_ply(self, path):
        
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)

        features_dc = np.zeros((xyz.shape[0], self.num_channels, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == self.num_channels * (self.max_sh_degree + 1) ** 2 - self.num_channels

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        features_extra = features_extra.reshape((features_extra.shape[0], \
                                                 self.num_channels, (self.max_sh_degree + 1) ** 2 - 1))

        atten_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("atten")]
        attenuations = np.zeros((xyz.shape[0], len(atten_names)))
        for idx, attr_name in enumerate(atten_names):
            attenuations[:, idx] = np.asarray(plydata.elements[0][attr_name])


        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])


        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz           = nn.Parameter(torch.tensor(xyz,            dtype=torch.float, device="cuda").requires_grad_(True))
        self._attenuation   = nn.Parameter(torch.tensor(attenuations,   dtype=torch.float, device="cuda").requires_grad_(True))

        self._features_dc   = nn.Parameter(torch.tensor(features_dc,    dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self._scaling       = nn.Parameter(torch.tensor(scales,         dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation      = nn.Parameter(torch.tensor(rots,           dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))

        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))

        l.append('atten')

        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))

        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        return l


    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        attenuations = self._attenuation.detach().cpu().numpy()

        scale = self._scaling.detach().cpu().numpy()

        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)  # (182686,)

        attributes = np.concatenate((xyz, normals, f_dc, f_rest, attenuations, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def training_setup(self, training_args):

        self.percent_dense = training_args.percent_dense
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz],           'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc],   'lr': training_args.feature_lr,                               "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0,                        "name": "f_rest"},
            {'params': [self._attenuation],   'lr': training_args.opacity_lr,                               "name": "attenuation"},
            {'params': [self._scaling],       'lr': training_args.scaling_lr,                               "name": "scaling"},
            {'params': [self._rotation],      'lr': training_args.rotation_lr,                              "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init       = training_args.position_lr_init  * self.spatial_lr_scale,
                                                    lr_final      = training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult = training_args.position_lr_delay_mult,
                                                    max_steps     = training_args.position_lr_max_steps)


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''

        for param_group in self.optimizer.param_groups:

            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                
                return lr


    def capture(self):

        return (self._xyz,
                self._features_dc,
                self._features_rest,
                # self._radiance,
                self._attenuation,
                self._scaling,
                self._rotation,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                )


    def restore(self, model_args, training_args):
        (self._xyz, 
        self._features_dc, 
         self._features_rest, 
         self._attenuation, 
         self._scaling, 
         self._rotation, 
         self.max_radii2D, 
         xyz_gradient_accum, 
         denom, 
         opt_dict, 
         self.spatial_lr_scale) = model_args

        self.training_setup(training_args)

        self.xyz_gradient_accum = xyz_gradient_accum

        self.denom = denom

        self.optimizer.load_state_dict(opt_dict)


    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


    def reset_attenuation(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_attenuation, torch.ones_like(self.get_attenuation) * 0.01))

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "attenuation")
        self._attenuation = optimizable_tensors["attenuation"]


    def replace_tensor_to_optimizer(self, tensor, name):

        optimizable_tensors = {}

        for group in self.optimizer.param_groups:

            if group["name"] == name:

                stored_state = self.optimizer.state.get(group['params'][0], None)

                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]

                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))

                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def _prune_optimizer(self, mask):
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:

            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def prune_points(self, mask):

        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        
        self._features_dc   = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]

        self._attenuation = optimizable_tensors["attenuation"]
        self._scaling     = optimizable_tensors["scaling"]
        self._rotation    = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom       = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:

            assert len(group["params"]) == 1

            extension_tensor = tensors_dict[group["name"]]   

            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], 
                                                     torch.zeros_like(extension_tensor)), dim=0)
                
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], 
                                                        torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]

                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], 
                                                             extension_tensor), dim=0).requires_grad_(True))

                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]

            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], 
                                                             extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def densification_postfix(self, 
                              new_xyz, 
                              new_features_dc, 
                              new_features_rest, 
                              new_attenuation, 
                              new_scaling, 
                              new_rotation):
        d = {
            "xyz":         new_xyz,
            "f_dc":        new_features_dc,
            "f_rest":      new_features_rest,
            "attenuation": new_attenuation,
            "scaling":     new_scaling,
            "rotation":    new_rotation
            }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]               
        self._features_dc = optimizable_tensors["f_dc"]      
        self._features_rest = optimizable_tensors["f_rest"]  

        self._attenuation = optimizable_tensors["attenuation"]       
        self._scaling = optimizable_tensors["scaling"]       
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):

        n_init_points = self.get_xyz.shape[0]  

        padded_grad = torch.zeros((n_init_points), device="cuda")

        padded_grad[:grads.shape[0]] = grads.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)

        means =torch.zeros((stds.size(0), 3), device="cuda")

        samples = torch.normal(mean=means, std=stds)

        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) \
            + self.get_xyz[selected_pts_mask].repeat(N, 1)
        
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))

        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)

        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)

        new_attenuation = self._attenuation[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, 
                                #    new_features,
                                   new_features_dc, 
                                   new_features_rest, 
                                   new_attenuation, 
                                   new_scaling, 
                                   new_rotation)

        prune_filter = torch.cat((selected_pts_mask, 
                                  torch.zeros(N * selected_pts_mask.sum(), 
                                              device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]

        new_scaling = self._scaling[selected_pts_mask]

        new_rotation = self._rotation[selected_pts_mask]

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]

        new_attenuation = self._attenuation[selected_pts_mask]


        self.densification_postfix(new_xyz, 
                                #    new_features,
                                   new_features_dc, 
                                   new_features_rest, 
                                   new_attenuation, 
                                   new_scaling, 
                                   new_rotation)


    def densify_and_prune(self, max_grad, min_attenuation, scene_extent, max_screen_size):
        
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, scene_extent)
        self.densify_and_split(grads, max_grad, scene_extent)

        prune_mask = (self.get_attenuation < min_attenuation).squeeze()  

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size 

            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * scene_extent    

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)


        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):

        self.xyz_gradient_accum[update_filter] += \
            torch.norm(viewspace_point_tensor.grad[update_filter, :3], dim=-1, keepdim=True)
        
        self.denom[update_filter] += 1

