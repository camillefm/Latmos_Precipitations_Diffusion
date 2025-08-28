import torch
def split_tensor_to_grid(tensor, tile_height, tile_width):
    """
    Splits a 3D tensor (C, H, W) into smaller tiles.

    Parameters:
    - tensor: 3D tensor or numpy array (C, H, W)
    - tile_height: height of each tile
    - tile_width: width of each tile

    Returns:
    - tiles: list of 3D tensors (C, tile_height, tile_width)
    - grid_shape: (num_rows, num_cols)
    """
    C, H, W = tensor.shape
    num_rows = H // tile_height
    num_cols = W // tile_width

    tiles = []
    for r in range(num_rows):
        for c in range(num_cols):
            top = r * tile_height
            left = c * tile_width
            tile = tensor[:, top:top + tile_height, left:left + tile_width]
            tiles.append(tile)

    return tiles, (num_rows, num_cols)


def merge_tiles_to_tensor(tiles, grid_shape, tile_height, tile_width):
    num_rows, num_cols = grid_shape
    if len(tiles) != num_rows * num_cols:
        raise ValueError(f"Number of tiles ({len(tiles)}) does not match grid shape ({num_rows} x {num_cols})")

    # Squeeze the first dimension if needed, assume all tiles have same shape
    first_tile = tiles[0]
    if first_tile.dim() == 4 and first_tile.shape[0] == 1:
        tiles = [t.squeeze(0) for t in tiles]

    C = tiles[0].shape[0]
    H = num_rows * tile_height
    W = num_cols * tile_width

    device = tiles[0].device
    dtype = tiles[0].dtype

    merged = torch.zeros((C, H, W), dtype=dtype, device=device)

    for idx, tile in enumerate(tiles):
        if tile.shape != (C, tile_height, tile_width):
            raise ValueError(f"Tile at index {idx} has shape {tile.shape} but expected {(C, tile_height, tile_width)}")
        r = idx // num_cols
        c = idx % num_cols
        top = r * tile_height
        left = c * tile_width
        merged[:, top:top + tile_height, left:left + tile_width] = tile

    return merged
