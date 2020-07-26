import torch
import torch.nn as nn

import torchvision.models as models

from layers import GCNLayer, GUnpooling

'''
Part of the code are adapted from
https://github.com/noahcao/Pixel2Mesh
'''

class GResNet(nn.Module):
    def __init__(self, in_feature, hidden_dim, graph, activation=None):
        super(GResNet, self).__init__()
        self.graph = graph
        self.conv1 = GCNLayer(in_features=in_feature, out_features=hidden_dim, dgl_g=self.graph)
        self.conv2 = GCNLayer(in_features=hidden_dim, out_features=in_feature, dgl_g=self.graph)
        self.activation = F.relu if activation else None

    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.activation:
            x = self.activation(x)
        x = self.conv2(x)
        if self.activation:
            x = self.activation(x)

        return (inputs + x) * 0.5

class MeshBlock(nn.Module):
    def __init__(self, num_blocks, in_feature, hidden_dim, out_feature, graph, activation=None):
        super(MeshBlock, self).__init__()

        resblock_layers = [GResNet(in_feature=hidden_dim, hidden_dim=hidden_dim, graph=graph, activation=activation)
                           for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*resblock_layers)
        self.conv1 = GCNLayer(in_features=in_feature, out_features=hidden_dim, graph=graph)
        self.conv2 = GCNLayer(in_features=hidden_dim, out_features=out_feature, graph=graph)
        self.activation = F.relu if activation else None

class Pixel2MeshModel(nn.Module):
    def __init__(self, hidden_dim, last_hidden_dim, coord_dim, ellipsoid, camera_f, camera_c, mesh_pos, gconv_activation):
        self.hidden_dim = hidden_dim
        self.coord_dim = coord_dim
        self.last_hidden_dim = last_hidden_dim
        self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)
        self.gconv_activation = options.gconv_activation

        # perpare the image side backbone
        # NOTE: how to use this? What if we load from ckpt?
        self.img_backbone = models.resnet50(pretrained=True)
        self.features_dim = self.img_backbone.features_dim + self.coord_dim

        # GCN for meshes
        self.mesh_backbone = nn.ModuleList([
            MeshBlock(6, self.features_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.dgl_graph[0], activation=self.gconv_activation),
            MeshBlock(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.dgl_graph[1], activation=self.gconv_activation),
            MeshBlock(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.last_hidden_dim,
                        ellipsoid.dgl_graph[2], activation=self.gconv_activation)
        ])

        # Graph Unpooling
        self.unpooling_layers = nn.ModuleList([
            GraphUnpooling(ellipsoid.unpool_idx[0]),
            GraphUnpooling(ellipsoid.unpool_idx[1])
        ])

        self.projection = GProjection(mesh_pos, camera_f, camera_c, bound=options.z_threshold,
                                      tensorflow_compatible=options.align_with_tensorflow)

        self.gconv = GCNLayer(in_features=self.last_hidden_dim, out_features=self.coord_dim,
                           graph=ellipsoid.dgl_g[2])


    def forward(self, img):
        batch_size = img.size(0)
        img_feats = self.nn_encoder(img)
        img_shape = self.projection.image_feature_shape(img)

        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)
        # GCN Block 1
        x = self.projection(img_shape, img_feats, init_pts)
        x1, x_hidden = self.gcns[0](x)

        # before deformation 2
        x1_up = self.unpooling[0](x1)

        # GCN Block 2
        x = self.projection(img_shape, img_feats, x1)
        x = self.unpooling[0](torch.cat([x, x_hidden], 2))
        # after deformation 2
        x2, x_hidden = self.gcns[1](x)

        # before deformation 3
        x2_up = self.unpooling[1](x2)

        # GCN Block 3
        x = self.projection(img_shape, img_feats, x2)
        x = self.unpooling[1](torch.cat([x, x_hidden], 2))
        x3, _ = self.gcns[2](x)
        if self.gconv_activation:
            x3 = F.relu(x3)
        # after deformation 3
        x3 = self.gconv(x3)

        return {
            "pred_coord": [x1, x2, x3],
            "pred_coord_before_deform": [init_pts, x1_up, x2_up],
        }