import torch
import torch.nn.functional as F

def generate_attention_bias_from_gaze_coords(
    gaze_coords: torch.Tensor,  # [B, T, 2] in pixel units (x, y)
    num_heads: int,
    image_size: int = 224,
    patch_size: int = 16,
    sigma: float = 1.0
):
    """
    Generate attention bias tensor from gaze coordinates.

    Args:
        gaze_coords: [B, T, 2], pixel coords (x, y) in [0, 1]
        num_heads: number of attention heads
        image_size: int, input image height/width (assumed square)
        patch_size: int, size of each patch (e.g., 16)
        sigma: float, std deviation for Gaussian in patch units

    Returns:
        attn_bias: [B*T, num_heads, patches^2 + 1, patches^2 + 1]
    """

    gaze_coords = gaze_coords* image_size  # Scale to pixel units
    gaze_coords = gaze_coords.round().long()

    B, T, _ = gaze_coords.shape
    device = gaze_coords.device

    patches = image_size // patch_size
    HW = patches * patches

    # Convert pixel coords to patch indices (x, y)
    # Clamp coords just in case
    x = gaze_coords[..., 0].clamp(0, image_size - 1)
    y = gaze_coords[..., 1].clamp(0, image_size - 1)

    patch_x = x / patch_size
    patch_y = y / patch_size

    centers = torch.stack([patch_y, patch_x], dim=-1).view(B * T, 2)  # [B*T, 2] (y, x)

    # Create grid of patch indices
    grid_y, grid_x = torch.meshgrid(torch.arange(patches, device=device), torch.arange(patches, device=device), indexing='ij')
    grid = torch.stack([grid_y, grid_x], dim=-1).view(HW, 2)  # [HW, 2]

    # Compute squared distances from gaze center to each patch
    dists = ((grid.unsqueeze(0) - centers.unsqueeze(1)) ** 2).sum(dim=-1)  # [B*T, HW]

    # Gaussian weights
    gauss = torch.exp(-dists / (2 * sigma ** 2))  # [B*T, HW]
    gauss = gauss / (gauss.sum(dim=1, keepdim=True) + 1e-8)

    # CLS token bias is zero
    cls_bias = torch.zeros(B * T, 1, device=device)

    bias_1d = torch.cat([cls_bias, gauss], dim=1)  # [B*T, HW + 1]

    # Reshape to [B*T, heads, HW + 1, HW + 1] with broadcasting on last dim
    bias_1d = bias_1d.unsqueeze(1).unsqueeze(2)  # [B*T, 1, 1, HW + 1]
    attn_bias = bias_1d.expand(-1, num_heads, HW + 1, -1)  # [B*T, heads, HW+1, HW+1]

    return attn_bias


def generate_aggregate_weight_from_gaze_coords(
    gaze_coords: torch.Tensor,  # [B, T, 2] in pixel units (x, y)
    grid_num: int = 14,
    sigma: float = 1.0
):
    """
    Generate attention bias tensor from gaze coordinates.

    Args:
        gaze_coords: [B, T, 2], pixel coords (x, y) in [0, 1]
        grid_num: int, number of grid cells in each dimension
        sigma: float, std deviation for Gaussian in patch units

    Returns:
        attn_bias: [B, T, grid_num, grid_num]
    """

    B, T, _ = gaze_coords.shape
    device = gaze_coords.device

    # Convert normalized gaze coords to grid indices
    gaze_grid = gaze_coords * grid_num
    gaze_grid = gaze_grid.clamp(0, grid_num - 1)

    # Create grid coordinate map
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_num, device=device),
        torch.arange(grid_num, device=device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).float()  # [grid_num, grid_num, 2]

    # Expand grid to [1, 1, grid_num, grid_num, 2]
    grid = grid.unsqueeze(0).unsqueeze(0)

    # Expand gaze coords to [B, T, 1, 1, 2]
    centers = gaze_grid.unsqueeze(2).unsqueeze(3)

    # Compute squared distances
    dists = ((grid - centers) ** 2).sum(dim=-1)  # [B, T, grid_num, grid_num]

    # Gaussian weights
    weights = torch.exp(-dists / (2 * sigma ** 2))  # [B, T, grid_num, grid_num]

    # Normalize spatially
    weights = weights / (weights.sum(dim=(-2, -1), keepdim=True) + 1e-8)

    return weights  # [B, T, grid_num, grid_num]