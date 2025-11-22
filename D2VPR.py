import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from backbone.vision_transformer_small import vit_small
import math


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.work_with_tokens = work_with_tokens

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1. / p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.dim() == 4:
            return x[:, :, 0, 0]
        elif x.dim() == 3:
            return x[:, :, 0]
        else:
            raise ValueError("Unsupported input dimension")


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class DeformablePositionEncoder(nn.Module):


    def __init__(self, feat_dim=768, pos_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, feat_dim)
        )

    def forward(self, features, deform_positions):

        B, N, D = features.shape
        device = features.device


        pos_list = []
        for pos in deform_positions:
            cx, cy = pos["center"]
            w, h = pos["size"]

            pos_tensor = torch.stack([cx, cy, w, h], dim=1)
            pos_list.append(pos_tensor)

        pos_tensor = torch.stack(pos_list, dim=1)  # [B, N, 4]


        pos_enc = self.fc(pos_tensor)


        return features + pos_enc





class HierarchicalDownTopFusion(nn.Module):


    def __init__(self, in_dim=768, hidden_dim=256):
        super().__init__()

        self.small_to_medium = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, in_dim)
        )


        self.medium_to_global = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, in_dim)
        )


        self.adjacency_mapping = {
            0: [0, 1, 3, 4],
            1: [1, 2, 4, 5],
            2: [3, 4, 6, 7],
            3: [4, 5, 7, 8],
            "global": [0, 1, 2, 3]
        }

    def forward(self, global_feat, medium_feats, small_feats):


        enhanced_medium = []
        for i, med_feat in enumerate(medium_feats):

            related_small = [small_feats[j] for j in self.adjacency_mapping[i]]


            small_pooled = torch.stack(related_small, dim=0).mean(dim=0)

            small_transformed = self.small_to_medium(small_pooled)
            enhanced = med_feat + small_transformed
            enhanced_medium.append(enhanced)


        related_medium = [enhanced_medium[j] for j in self.adjacency_mapping["global"]]
        medium_pooled = torch.stack(related_medium, dim=0).mean(dim=0)


        medium_transformed = self.medium_to_global(medium_pooled)
        enhanced_global = global_feat + medium_transformed

        return enhanced_global


class D2VPR(nn.Module):
    def __init__(self, pretrained_foundation=False, foundation_model_path=None,cross_image_encoder=True):
        super().__init__()
        self.backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.aggregation = nn.Sequential(L2Norm(), GeM(work_with_tokens=None), Flatten())
        self.cross_image_encoder=cross_image_encoder

        self.offset_generator = nn.Sequential(
            nn.Conv2d(768 * 2, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 4, kernel_size=1)  
        )
        self.topdown_fusion = HierarchicalDownTopFusion(in_dim=768, hidden_dim=256)

        self.pos_encoder = DeformablePositionEncoder(feat_dim=768, pos_dim=128)

        if cross_image_encoder:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=768, nhead=16, dim_feedforward=2048,
                activation="gelu", dropout=0.1, batch_first=False
            )

            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)


        self.medium_regions = [
            {"name": "x10", "roi": (0.0, 0.0, 0.5, 0.5), "size": (8, 8)},
            {"name": "x11", "roi": (0.5, 0.0, 1.0, 0.5), "size": (8, 8)},
            {"name": "x12", "roi": (0.0, 0.5, 0.5, 1.0), "size": (8, 8)},
            {"name": "x13", "roi": (0.5, 0.5, 1.0, 1.0), "size": (8, 8)}
        ]

        self.small_regions = [
            {"name": "x20", "roi": (0.0, 0.0, 5 / 16, 5 / 16), "size": (5, 5)},
            {"name": "x21", "roi": (5 / 16, 0.0, 11 / 16, 5 / 16), "size": (5, 6)},
            {"name": "x22", "roi": (11 / 16, 0.0, 1.0, 5 / 16), "size": (5, 5)},
            {"name": "x23", "roi": (0.0, 5 / 16, 5 / 16, 11 / 16), "size": (6, 5)},
            {"name": "x24", "roi": (5 / 16, 5 / 16, 11 / 16, 11 / 16), "size": (6, 6)},
            {"name": "x25", "roi": (11 / 16, 5 / 16, 1.0, 11 / 16), "size": (6, 5)},
            {"name": "x26", "roi": (0.0, 11 / 16, 5 / 16, 1.0), "size": (5, 5)},
            {"name": "x27", "roi": (5 / 16, 11 / 16, 11 / 16, 1.0), "size": (5, 6)},
            {"name": "x28", "roi": (11 / 16, 11 / 16, 1.0, 1.0), "size": (5, 5)}
        ]


        self.medium_centers = []
        for region in self.medium_regions:
            x1, y1, x2, y2 = region["roi"]
            self.medium_centers.append((
                (x1 + x2) / 2,
                (y1 + y2) / 2
            ))

    def deformable_pooling(self, full_feat, global_feat, roi, output_size):
        B, D, H_full, W_full = full_feat.shape
        device = full_feat.device
        x1, y1, x2, y2 = roi

        # Calculate the width and height of the ROI region
        roi_width = x2 - x1
        roi_height = y2 - y1

        # Generate the basic sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(y1, y2, output_size[0], device=device),
            torch.linspace(x1, x2, output_size[1], device=device),
            indexing='ij'  
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1)
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)

        # Generate offsets and scaling factors
        global_feat_expanded = global_feat.view(B, D, 1, 1).expand(-1, -1, H_full, W_full)
        fused_feat = torch.cat([full_feat, global_feat_expanded], dim=1)
        offset_map = self.offset_generator(fused_feat)  # [B, 4, H_full, W_full]

        # Sample the offset of the ROI region from the offset map. Here, we do not normalize the base_grid, because the offset generator can learn this coordinate transformation.
        offset_map = offset_map.permute(0, 2, 3, 1)  

        roi_offset = F.grid_sample(
            offset_map.permute(0, 3, 1, 2),  
            base_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).permute(0, 2, 3, 1)  # [B, H_out, W_out, 4]
        dxdy = roi_offset[..., :2]  
        log_swsh = roi_offset[..., 2:]  
        swsh = 0.5 + 1.5 * torch.sigmoid(log_swsh)
        max_offset = 0.25
        dxdy = torch.clamp(dxdy, min=-max_offset, max=max_offset)

        # Apply the scaling factors to the base grid
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        rel_x = (base_grid[..., 0] - center_x) / (roi_width / 2)
        rel_y = (base_grid[..., 1] - center_y) / (roi_height / 2)
        scaled_x = rel_x * swsh[..., 0]
        scaled_y = rel_y * swsh[..., 1]
        abs_x = center_x + scaled_x * (roi_width / 2)
        abs_y = center_y + scaled_y * (roi_height / 2)

        # Apply positional offsets
        deform_x = abs_x + dxdy[..., 0] * roi_width
        deform_y = abs_y + dxdy[..., 1] * roi_height

        # Ensure the coordinates are within the [0, 1] range
        deform_x = torch.clamp(deform_x, 0, 1)
        deform_y = torch.clamp(deform_y, 0, 1)

        # normalize
        deform_grid = torch.stack((deform_x, deform_y), dim=-1)
        deform_grid = deform_grid * 2 - 1

        # Sampling
        warped_feat = F.grid_sample(
            full_feat,
            deform_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        # Calculate the actual position information after deformation
        center_x_deform = deform_x.mean(dim=(1, 2))
        center_y_deform = deform_y.mean(dim=(1, 2))
        width_deform = deform_x.amax(dim=(1, 2)) - deform_x.amin(dim=(1, 2))
        height_deform = deform_y.amax(dim=(1, 2)) - deform_y.amin(dim=(1, 2))

        deform_position = {
            "center": (center_x_deform, center_y_deform),
            "size": (width_deform, height_deform)
        }

        return self.aggregation(warped_feat), deform_position

    def forward(self, x):

        x = self.backbone(x)
        B, P, D = x["x_prenorm"].shape
        W = H = int(math.sqrt(P - 1))
        x0 = x["x_norm_clstoken"]
        x_p = x["x_norm_patchtokens"].view(B, W, H, D).permute(0, 3, 1, 2)  # [B, D, 16, 16]


        global_center_x = torch.full((B,), 0.5, device=x0.device)
        global_center_y = torch.full((B,), 0.5, device=x0.device)
        global_w = torch.full((B,), 1.0, device=x0.device)
        global_h = torch.full((B,), 1.0, device=x0.device)
        global_position = {
            "center": (global_center_x, global_center_y),
            "size": (global_w, global_h)
        }


        medium_features = []
        medium_positions = []

        for region in self.medium_regions:
            pooled, deform_pos = self.deformable_pooling(
                x_p,
                x0,
                region["roi"],
                region["size"]
            )
            medium_features.append(pooled)
            medium_positions.append(deform_pos)


        small_features = []
        small_positions = []

        for region in self.small_regions:
            pooled, deform_pos = self.deformable_pooling(
                x_p,
                x0,
                region["roi"],
                region["size"]
            )
            small_features.append(pooled)
            small_positions.append(deform_pos)


        enhanced_global = self.topdown_fusion(x0, medium_features, small_features)
        all_features = [enhanced_global.unsqueeze(1)]
        all_positions = [global_position]

        for feat in medium_features:
            all_features.append(feat.unsqueeze(1))
        all_positions.extend(medium_positions)

        for feat in small_features:
            all_features.append(feat.unsqueeze(1))
        all_positions.extend(small_positions)

        all_features = torch.cat(all_features, dim=1)


        all_features = self.pos_encoder(all_features, all_positions)

        if self.cross_image_encoder:

            x_transformed = self.encoder(all_features)
        else:
            x_transformed=all_features


        x = x_transformed.reshape(B, -1)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)

        return x


def get_backbone(pretrained_foundation, foundation_model_path):
    backbone = vit_small(patch_size=14, img_size=518, init_values=1, block_chunks=0)
    if pretrained_foundation:
        assert foundation_model_path is not None, "Please specify foundation model path."
        backbone.load_state_dict(torch.load('./dinov2_vits14_pretrain.pth'), strict=False)

    return backbone