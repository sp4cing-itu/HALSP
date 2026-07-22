# halsp.py
# HALSP-Net: Hierarchical Active channel Latent Shared Projection Network
# Reference: "HALSP-Net: A Shared Projection Architecture with Dynamic Channel Selection"
# This module implements the core HalspNetStage and a ResNet50 wrapper.
#
# The stage uses a single learnable weight matrix (master_weight) to perform three roles:
#   1. Entry projection (C_in -> C_mid)
#   2. Inner channel mixing (C_mid -> C_out, cyclic column shift per block)
#   3. Exit projection (C_mid -> C_out)
# Channel mixing and spatial processing are decoupled: depthwise convolutions handle the
# spatial part. During training a dynamic subset of latent channels is active; at inference
# the dense weight matrix is used.

import torch
import torch.nn as nn
import torch.nn.functional as F


class HalspNetStage(nn.Module):
    """
    A single HALSP stage containing multiple blocks that share the same projection matrix.
    
    Args:
        in_channels: number of input channels to the stage
        out_channels: number of output channels of the stage
        mid_channels: latent (bottleneck) channel dimension C_mid
        num_blocks: number of inner spatial-mixing blocks
        stride: stride for the entry convolution
        focus_ratio: fraction f_s of mid_channels forming the Focus Pool (default: 0.10)
        exploit_ratio: fraction D of Focus Pool channels used for exploitation (default: 0.80)
        explore_ratio: fraction ε of mid_channels used for exploration (default: 0.01)
        gn_groups: group normalization groups (used for alignment; >1 enables alignment)
        kernel_size: kernel size for depthwise convolutions
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels,
        num_blocks,
        stride,
        focus_ratio=0.10,
        exploit_ratio=0.80,
        explore_ratio=0.01,
        gn_groups=1,
        kernel_size=5,
    ):
        super().__init__()

        pad = kernel_size // 2

        # Base ratios (original settings) kept for later resetting via set_phase
        self.base_focus_ratio = focus_ratio
        self.base_exploit_ratio = exploit_ratio
        self.base_explore_ratio = explore_ratio

        # Dynamic ratios that can be changed during training phases
        self.dynamic_focus_ratio = focus_ratio
        self.dynamic_exploit_ratio = exploit_ratio
        self.dynamic_explore_ratio = explore_ratio

        self.gn_groups = gn_groups
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.stride = stride
        self._pad = pad

        # Column shifts for the inner mixing: each block uses a shifted version of the master weight
        self._col_shifts = [
            (i * mid_channels) % out_channels for i in range(num_blocks)
        ]

        print(
            f"[DEBUG] HalspNetStage: {num_blocks} Block | "
            f"Matrix: {mid_channels}x{out_channels} (Shared)"
        )

        # The single shared weight matrix W (size: mid_channels x out_channels x 1 x 1)
        self.master_weight = nn.Parameter(
            torch.empty(mid_channels, out_channels, 1, 1)
        )
        nn.init.kaiming_normal_(
            self.master_weight, mode="fan_out", nonlinearity="relu"
        )

        # How often (in steps) the topology (active channel set) is refreshed
        self.update_freq = 10
        self.current_step = 0
        self.ema_momentum = 0.01   # for running input variance estimate

        # Caches for dynamic channel selection
        self.active_pool_cache = None   # Focus Pool indices (high-score channels)
        self.dead_pool_cache = None     # Reserve Pool indices
        self.cached_indices = None      # current active indices
        self._cached_col_compact = None # precomputed inner weight indices for sparse path
        self._cached_dw_idx = None      # precomputed depthwise weight indices for sparse path

        # Depthwise spatial filters: one per block per channel (grouped)
        self.dw_weight = nn.Parameter(
            torch.empty(num_blocks * mid_channels, 1, kernel_size, kernel_size)
        )
        for i in range(num_blocks):
            s = i * mid_channels
            e = s + mid_channels
            nn.init.kaiming_normal_(
                self.dw_weight.data[s:e], mode="fan_out", nonlinearity="relu"
            )
            # add slight noise to encourage diversity
            self.dw_weight.data[s:e].mul_(
                1.0 + torch.randn(mid_channels, 1, kernel_size, kernel_size) * 0.02
            )

        # Start indices for each block's depthwise weight slice
        self._dw_block_starts = [i * mid_channels for i in range(num_blocks)]

        # Exit batch norm
        self.exit_bn = nn.BatchNorm2d(out_channels)

        # Buffer used for sparse depthwise index calculation
        self.register_buffer(
            "_dw_offsets",
            torch.arange(num_blocks, dtype=torch.long) * mid_channels,
            persistent=False,
        )

        # Channel expansion: if in_channels != out_channels, we concatenate extra features
        self.main_path_upsampler = None
        if in_channels != out_channels:
            extra_channels = out_channels - in_channels
            self.main_path_upsampler = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    extra_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=pad,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(extra_channels),
            )

        # Downsampling for the skip connection when stride != 1
        self.downsample_path = None
        if stride != 1:
            self.downsample_path = nn.Sequential(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad,
                    groups=out_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        # Precompute the inner projection column indices for the dense path
        base_cols = torch.arange(mid_channels)
        inner_cols = []
        for i in range(num_blocks):
            shift = (i * mid_channels) % out_channels
            inner_cols.append((base_cols + shift) % out_channels)
        self.register_buffer(
            "_all_inner_cols", torch.stack(inner_cols), persistent=False
        )

        # Running estimate of input variance trace v (used in opportunity map)
        self.register_buffer("running_input_var", torch.zeros(out_channels))

    @staticmethod
    def _empty_long(device):
        """Return an empty long tensor on the given device."""
        return torch.empty(0, device=device, dtype=torch.long)

    @torch.no_grad()
    def run_topology_maintenance(self, external_momentum_map=None):
        """
        Recompute the Focus and Reserve Pools based on importance scores.
        If an external momentum map is provided (from optimizer), the importance is
        mean(|W * momentum|). Otherwise it's mean(|W|).
        """
        w = self.master_weight.detach()

        if (
            external_momentum_map is not None
            and id(self.master_weight) in external_momentum_map
        ):
            opt_momentum = external_momentum_map[id(self.master_weight)]
            importance_map = (w * opt_momentum).abs()
            scores = importance_map.mean(dim=(1, 2, 3))
            if scores.sum() < 1e-9:
                scores = w.abs().mean(dim=(1, 2, 3))
        else:
            scores = w.abs().mean(dim=(1, 2, 3))

        num_focus_pool = int(self.mid_channels * self.dynamic_focus_ratio)
        _, sorted_idx = torch.sort(scores, descending=True)

        self.active_pool_cache = sorted_idx[:num_focus_pool]   # Focus Pool
        self.dead_pool_cache = sorted_idx[num_focus_pool:]     # Reserve Pool
        # Immediately update the sparse cache with the new active set
        self._update_sparse_cache(self.get_focus_indices())

    @torch.no_grad()
    def set_strategy(
        self,
        new_exploit_ratio=None,
        new_explore_ratio=None,
        new_focus_ratio=None,
    ):
        """
        Dynamically change the sparsity ratios and clear relevant caches.
        Used when switching between warmup/search/cooldown phases.
        """
        if new_exploit_ratio is not None:
            self.dynamic_exploit_ratio = new_exploit_ratio
        if new_explore_ratio is not None:
            self.dynamic_explore_ratio = new_explore_ratio
        if new_focus_ratio is not None:
            self.dynamic_focus_ratio = new_focus_ratio

        # Force re‑evaluation of indices on next forward
        self.cached_indices = None
        self._cached_col_compact = None
        self._cached_dw_idx = None
        self.active_pool_cache = None
        self.dead_pool_cache = None

    @torch.no_grad()
    def _update_sparse_cache(self, active_idx):
        """Precompute index tensors for the sparse forward pass."""
        self.cached_indices = active_idx
        out_ch = self.out_channels

        # For each block, the column indices in the weight matrix (cyclic shift)
        self._cached_col_compact = torch.stack(
            [(active_idx + s) % out_ch for s in self._col_shifts]
        )

        # Depthwise weight indices for the active channels across all blocks
        self._cached_dw_idx = (
            active_idx.unsqueeze(0) + self._dw_offsets.unsqueeze(1)
        ).reshape(-1)

    @torch.no_grad()
    def get_focus_indices(self):
        """
        Determine the active channel slice for this training step.
        Combines exploit (uniform from Focus Pool) and explore (from Reserve Pool
        using opportunity map Q) channels. The final set is sorted and optionally
        aligned to group norm groups.
        """
        if self.active_pool_cache is None:
            # Initialise pools if not yet created
            self.run_topology_maintenance(external_momentum_map=None)

        w = self.master_weight.detach()
        device = w.device
        active_pool = self.active_pool_cache   # Focus Pool
        dead_pool = self.dead_pool_cache       # Reserve Pool

        # ----- Exploit: uniform sample from Focus Pool -----
        n_exploit = int(len(active_pool) * self.dynamic_exploit_ratio)
        if len(active_pool) > 0:
            perm_active = torch.randperm(len(active_pool), device=device)
            exploit_idx = active_pool[perm_active[:n_exploit]]
        else:
            exploit_idx = self._empty_long(device)

        # ----- Explore: select channels from Reserve Pool via opportunity map -----
        n_explore = int(self.mid_channels * self.dynamic_explore_ratio)
        if len(dead_pool) > 0:
            n_explore = min(n_explore, len(dead_pool))

            if n_explore > 0 and self.current_step > 0:
                # Compute opportunity map Q[d] = v[d] / (K[d] + ε)
                # where K[d] is the sum of absolute weights of active channels along dimension d
                active_slice = torch.index_select(w, 0, active_pool)
                dead_slice = torch.index_select(w, 0, dead_pool)

                active_coverage = active_slice.view(len(active_pool), -1).abs()
                coverage_per_dim = active_coverage.sum(dim=0)
                opportunity = self.running_input_var / (coverage_per_dim + 1e-8)

                # Cosine similarity between dead channel weights and opportunity vector
                dead_w = dead_slice.view(len(dead_pool), -1).abs()
                dead_w_norm = F.normalize(dead_w, dim=1)
                opp_norm = F.normalize(opportunity, dim=0)
                scores = torch.matmul(dead_w_norm, opp_norm)
                scores = scores.add_(1e-8)
                probs = scores / scores.sum()
                chosen = torch.multinomial(probs, n_explore, replacement=False)
                explore_idx = dead_pool[chosen]

                # Free memory
                del (
                    active_slice,
                    dead_slice,
                    active_coverage,
                    dead_w,
                    dead_w_norm,
                    opp_norm,
                    scores,
                    probs,
                    opportunity,
                )
            elif n_explore > 0:
                # Fallback: uniform random explore
                perm_dead = torch.randperm(len(dead_pool), device=device)
                explore_idx = dead_pool[perm_dead[:n_explore]]
            else:
                explore_idx = self._empty_long(device)
        else:
            explore_idx = self._empty_long(device)

        # Combine and sort
        final_indices = torch.sort(
            torch.cat((exploit_idx, explore_idx), dim=0)
        )[0]

        # Align to group norm groups if needed
        if self.gn_groups > 1:
            n = len(final_indices)
            n_aligned = (n // self.gn_groups) * self.gn_groups
            if n_aligned == 0:
                if len(active_pool) >= self.gn_groups:
                    final_indices = active_pool[: self.gn_groups]
                else:
                    final_indices = torch.arange(
                        self.gn_groups, device=device, dtype=torch.long
                    )
            else:
                final_indices = final_indices[:n_aligned]

        return final_indices

    # ------------------------------------------------------------------ #
    #  SPARSE FORWARD                                                     #
    # ------------------------------------------------------------------ #
    def _forward_sparse(self, x_expanded, active_idx):
        """
        Forward pass using only the active subset of channels.
        Entry projection, inner mixing with cyclic column shifts, and exit projection
        all share the same master_weight restricted to active_idx.
        """
        num_active = len(active_idx)
        pad = self._pad
        num_blocks = self.num_blocks
        out_ch = self.out_channels

        # Entry projection: C_in -> C_mid (only active channels)
        w_entry = torch.index_select(self.master_weight, 0, active_idx)
        latent = F.conv2d(x_expanded, w_entry, stride=self.stride, padding=0)

        # Depthwise weights for all blocks, restricted to active channels
        dw_all = torch.index_select(self.dw_weight, 0, self._cached_dw_idx)
        dw_weights = dw_all.split(num_active)

        # Precompute the inner projection weight slices for each block
        w_2d = w_entry.view(num_active, out_ch)
        w_exp = w_2d.unsqueeze(0).expand(num_blocks, -1, -1)
        idx_exp = self._cached_col_compact.unsqueeze(1).expand(
            -1, num_active, -1
        )
        w_all = torch.gather(w_exp, 2, idx_exp).unsqueeze(-1).unsqueeze(-1)

        # Inner blocks: depthwise conv -> inner projection -> residual add
        for i in range(num_blocks):
            out = F.gelu(latent)
            out = F.conv2d(
                out, dw_weights[i], stride=1, padding=pad, groups=num_active
            )
            out = F.conv2d(out, w_all[i], stride=1, padding=0)
            latent = out.add_(latent)

        # Exit projection: C_mid -> C_out (active -> full)
        w_exit = w_entry.permute(1, 0, 2, 3).contiguous()
        out = F.conv2d(latent, w_exit, stride=1, padding=0)
        return self.exit_bn(out)

    # ------------------------------------------------------------------ #
    #  DENSE FORWARD                                                      #
    # ------------------------------------------------------------------ #
    def _forward_dense(self, x_expanded):
        """
        Forward pass using the full dense master_weight.
        Used at inference time.
        """
        w_entry = self.master_weight
        num_blocks = self.num_blocks
        mid_ch = self.mid_channels
        pad = self._pad
        dw_weight = self.dw_weight
        block_starts = self._dw_block_starts

        # Entry projection
        latent = F.conv2d(x_expanded, w_entry, stride=self.stride, padding=0)

        # Precompute inner projection weight slices (cyclic shifts)
        w_2d = w_entry.view(mid_ch, self.out_channels)
        w_exp = w_2d.unsqueeze(0).expand(num_blocks, -1, -1)
        idx_exp = self._all_inner_cols.unsqueeze(1).expand(-1, mid_ch, -1)
        w_all = torch.gather(w_exp, 2, idx_exp).unsqueeze(-1).unsqueeze(-1)

        for i in range(num_blocks):
            out = F.gelu(latent)
            out = F.conv2d(
                out,
                dw_weight[block_starts[i] : block_starts[i] + mid_ch],
                stride=1,
                padding=pad,
                groups=mid_ch,
            )
            out = F.conv2d(out, w_all[i], stride=1, padding=0)
            latent = out.add_(latent)

        # Exit projection
        w_exit = w_entry.permute(1, 0, 2, 3).contiguous()
        out = F.conv2d(latent, w_exit, stride=1, padding=0)
        return self.exit_bn(out)

    # ------------------------------------------------------------------ #
    def forward(self, x):
        """
        Stage forward pass. If in training mode, performs sparse forward with the
        dynamically selected active channels; periodically updates the channel
        topology. In eval mode, uses the dense forward path.
        """
        # Handle channel expansion for the main path
        upsampler = self.main_path_upsampler
        if upsampler is not None:
            x_expanded = torch.cat((upsampler(x), x), dim=1)
        else:
            x_expanded = x

        # Downsampling for the skip connection
        downsample = self.downsample_path
        if downsample is not None:
            identity_global = downsample(x_expanded)
        else:
            identity_global = x_expanded

        if self.training:
            self.current_step += 1

            # Check if we need to recompute the active channel set
            should_update_topo = (
                self.cached_indices is None
                or (self.current_step - 1) % self.update_freq == 0
            )

            if should_update_topo:
                with torch.no_grad():
                    input_var = x_expanded.var(dim=(0, 2, 3))
                    self.running_input_var.lerp_(input_var, self.ema_momentum)

                active_idx = self.get_focus_indices()
                self._update_sparse_cache(active_idx)

            out = self._forward_sparse(x_expanded, self.cached_indices)
        else:
            out = self._forward_dense(x_expanded)

        # Residual connection
        out.add_(identity_global)
        return F.gelu(out)


class StandardBottleneck(nn.Module):
    """
    Classic ResNet bottleneck block (used for baseline comparisons, not in HALSP stages).
    """
    expansion = 4

    def __init__(
        self,
        in_channels,
        mid_channels,
        stride=1,
        downsample=None,
        groups=32,
        width_per_group=8,
    ):
        super().__init__()

        out_channels = mid_channels * self.expansion
        width = mid_channels * width_per_group * groups // 64

        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            width, width, 3, stride, 1, groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out.add_(identity)
        return self.relu(out)


class ResNet50(nn.Module):
    """
    ResNet50‑style architecture where all bottleneck stages are replaced by HALSP HalspNetStage.
    Supports warm‑up (dense), search (sparse), and cooldown (dense) phases.
    """
    def __init__(self, num_classes=1000):
        super().__init__()

        self.in_channels = 64
        self.opnet_stages = nn.ModuleList()   # keep track of HALSP stages for topology maintenance
        
        # Stem for CIFAR‑size inputs (32x32) – two 3x3 convs, no pooling
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # STANDART RESNET STEM FOR 224X224 IMAGENET SIZE PERFORMANCE COMPARISON
        """
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        """

        # Four HALSP stages matching ResNet50 block counts
        self.layer1 = self._make_layer(
            64, 3, 1,
            use_opnet_structure=True, kernel_size=3,
            focus_ratio=1.0, exploit_ratio=1.0, explore_ratio=0.0,  # fully dense
        )
        self.layer2 = self._make_layer(
            128, 4, 2,
            use_opnet_structure=True, kernel_size=3,
            focus_ratio=1.0, exploit_ratio=1.0, explore_ratio=0.0,  # fully dense
        )
        self.layer3 = self._make_layer(
            256, 6, 2,
            use_opnet_structure=True, kernel_size=3,
            focus_ratio=0.5, exploit_ratio=0.9, explore_ratio=0.05,  # sparse (ra=0.5)
        )
        self.layer4 = self._make_layer(
            512, 3, 2,
            use_opnet_structure=True, kernel_size=3,
            focus_ratio=0.5, exploit_ratio=0.9, explore_ratio=0.05,  # sparse (ra=0.5)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(
        self,
        mid_channels,
        blocks,
        stride,
        use_opnet_structure=False,
        focus_ratio=0.50,
        exploit_ratio=0.9,
        explore_ratio=0.05,
        gn_groups=1,
        kernel_size=5,
        groups=32,
        width_per_group=8,
    ):
        """
        Create a layer composed of either HALSP HalspNetStages or StandardBottlenecks.
        """
        expansion = 4

        if use_opnet_structure:
            out_channels = mid_channels * expansion
            stage = HalspNetStage(
                self.in_channels,
                out_channels,
                mid_channels,
                blocks,
                stride,
                focus_ratio,
                exploit_ratio,
                explore_ratio,
                gn_groups,
                kernel_size,
            )
            self.in_channels = out_channels
            self.opnet_stages.append(stage)
            return stage

        # Fallback: standard bottleneck blocks
        out_channels_std = mid_channels * expansion
        downsample = None
        if stride != 1 or self.in_channels != out_channels_std:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels_std, 1, stride, bias=False
                ),
                nn.BatchNorm2d(out_channels_std),
            )

        layers = [
            StandardBottleneck(
                self.in_channels,
                mid_channels,
                stride,
                downsample,
                groups=groups,
                width_per_group=width_per_group,
            )
        ]
        self.in_channels = out_channels_std

        for _ in range(1, blocks):
            layers.append(
                StandardBottleneck(
                    self.in_channels,
                    mid_channels,
                    groups=groups,
                    width_per_group=width_per_group,
                )
            )

        return nn.Sequential(*layers)

    def run_topology_maintenance(self, external_momentum_map=None):
        """Propagate topology maintenance to all HALSP stages."""
        for stage in self.opnet_stages:
            stage.run_topology_maintenance(external_momentum_map)

    def set_phase(self, phase="search"):
        """
        Switch training phase:
          - "warmup" / "cooldown": dense mode (focus_ratio=1.0, exploit=1.0, explore=0.0)
          - "search": use the base sparsity ratios of each stage
        """
        if phase in ("warmup", "cooldown"):
            for stage in self.opnet_stages:
                stage.set_strategy(
                    new_focus_ratio=1.0,
                    new_exploit_ratio=1.0,
                    new_explore_ratio=0.0,
                )
        elif phase == "search":
            for stage in self.opnet_stages:
                stage.set_strategy(
                    new_focus_ratio=stage.base_focus_ratio,
                    new_exploit_ratio=stage.base_exploit_ratio,
                    new_explore_ratio=stage.base_explore_ratio,
                )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # Quick parameter count check
    model = ResNet50(num_classes=10)
    print("---------------------------------------------------------")
    print(
        f"DEBUG: Param Count: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M"
    )
    print("---------------------------------------------------------")
