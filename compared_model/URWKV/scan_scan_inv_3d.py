import torch

def _get_dims(input_tensor):
    """Helper to get dimensions for 3D tensor."""
    return input_tensor.shape

def _get_original_dims(original_shape):
    """Helper to get dimensions from original_shape tuple."""
    return original_shape

# --- 1. 沿宽度 (W) 方向扫描 (左到右 / 右到左) ---

def scan_left_to_right(input_tensor):
    """
    对输入张量进行沿宽度 (W) 方向的从左到右扫描。
    输入形状: (B, C, D, H, W)
    变换后形状: (B, D * H * W, C)
    """
    B, C, D, H, W = _get_dims(input_tensor)
    # 调整维度顺序，使 W 成为最内层空间维度，C 移到最后
    # (B, C, D, H, W) -> (B, D, H, W, C)
    temp_tensor = input_tensor.permute(0, 2, 3, 4, 1)
    # 展平空间维度
    flattened = temp_tensor.reshape(B, D * H * W, C)
    return flattened

def scan_left_to_right_inv(transformed_tensor, original_shape):
    """
    逆变换: 复原 scan_left_to_right 变换后的数据。
    输入形状: (B, D * H * W, C)
    复原形状: (B, C, D, H, W)
    """
    B, C, D, H, W = _get_original_dims(original_shape)
    # 恢复空间维度
    # (B, D*H*W, C) -> (B, D, H, W, C)
    recovered = transformed_tensor.view(B, D, H, W, C)
    # 恢复原始维度顺序
    # (B, D, H, W, C) -> (B, C, D, H, W)
    return recovered.permute(0, 4, 1, 2, 3)

def scan_right_to_left(input_tensor):
    """
    对输入张量进行沿宽度 (W) 方向的从右到左扫描。
    输入形状: (B, C, D, H, W)
    变换后形状: (B, D * H * W, C)
    """
    B, C, D, H, W = _get_dims(input_tensor)
    # 首先沿宽度维度进行翻转
    flipped_tensor = torch.flip(input_tensor, dims=[-1]) # 翻转 W 维度
    # 调整维度顺序，使 W 成为最内层空间维度，C 移到最后
    # (B, C, D, H, W) -> (B, D, H, W, C)
    temp_tensor = flipped_tensor.permute(0, 2, 3, 4, 1)
    # 展平空间维度
    flattened = temp_tensor.reshape(B, D * H * W, C)
    return flattened

def scan_right_to_left_inv(transformed_tensor, original_shape):
    """
    逆变换: 复原 scan_right_to_left 变换后的数据。
    输入形状: (B, D * H * W, C)
    复原形状: (B, C, D, H, W)
    """
    B, C, D, H, W = _get_original_dims(original_shape)
    # 恢复空间维度
    # (B, D*H*W, C) -> (B, D, H, W, C)
    recovered = transformed_tensor.view(B, D, H, W, C)
    # 恢复原始维度顺序
    # (B, D, H, W, C) -> (B, C, D, H, W)
    recovered = recovered.permute(0, 4, 1, 2, 3)
    # 最后沿宽度维度进行翻转，抵消之前的翻转
    return torch.flip(recovered, dims=[-1])


# --- 2. 沿高度 (H) 方向扫描 (上到下 / 下到上) ---

def scan_up_to_down(input_tensor):
    """
    对输入张量进行沿高度 (H) 方向的从上到下扫描。
    输入形状: (B, C, D, H, W)
    变换后形状: (B, D * H * W, C)
    """
    B, C, D, H, W = _get_dims(input_tensor)
    # 调整维度顺序，使 H 成为最内层空间维度，C 移到最后
    # (B, C, D, H, W) -> (B, D, W, H, C)
    temp_tensor = input_tensor.permute(0, 2, 4, 3, 1)
    # 展平空间维度
    flattened = temp_tensor.reshape(B, D * W * H, C)
    return flattened

def scan_up_to_down_inv(transformed_tensor, original_shape):
    """
    逆变换: 复原 scan_up_to_down 变换后的数据。
    输入形状: (B, D * H * W, C)
    复原形状: (B, C, D, H, W)
    """
    B, C, D, H, W = _get_original_dims(original_shape)
    # 恢复空间维度
    # (B, D*W*H, C) -> (B, D, W, H, C)
    recovered = transformed_tensor.view(B, D, W, H, C)
    # 恢复原始维度顺序
    # (B, D, W, H, C) -> (B, C, D, H, W)
    return recovered.permute(0, 4, 1, 3, 2)

def scan_down_to_up(input_tensor):
    """
    对输入张量进行沿高度 (H) 方向的从下到上扫描。
    输入形状: (B, C, D, H, W)
    变换后形状: (B, D * H * W, C)
    """
    B, C, D, H, W = _get_dims(input_tensor)
    # 首先沿高度维度进行翻转
    flipped_tensor = torch.flip(input_tensor, dims=[-2]) # 翻转 H 维度
    # 调整维度顺序，使 H 成为最内层空间维度，C 移到最后
    # (B, C, D, H, W) -> (B, D, W, H, C)
    temp_tensor = flipped_tensor.permute(0, 2, 4, 3, 1)
    # 展平空间维度
    flattened = temp_tensor.reshape(B, D * W * H, C)
    return flattened

def scan_down_to_up_inv(transformed_tensor, original_shape):
    """
    逆变换: 复原 scan_down_to_up 变换后的数据。
    输入形状: (B, D * H * W, C)
    复原形状: (B, C, D, H, W)
    """
    B, C, D, H, W = _get_original_dims(original_shape)
    # 恢复空间维度
    # (B, D*W*H, C) -> (B, D, W, H, C)
    recovered = transformed_tensor.view(B, D, W, H, C)
    # 恢复原始维度顺序
    # (B, D, W, H, C) -> (B, C, D, H, W)
    recovered = recovered.permute(0, 4, 1, 3, 2)
    # 最后沿高度维度进行翻转，抵消之前的翻转
    return torch.flip(recovered, dims=[-2])


# --- 3. 沿深度 (D) 方向扫描 (前到后 / 后到前) ---

def scan_front_to_back(input_tensor):
    """
    对输入张量进行沿深度 (D) 方向的从前到后扫描。
    输入形状: (B, C, D, H, W)
    变换后形状: (B, D * H * W, C)
    """
    B, C, D, H, W = _get_dims(input_tensor)
    # 调整维度顺序，使 D 成为最内层空间维度，C 移到最后
    # (B, C, D, H, W) -> (B, H, W, D, C)
    temp_tensor = input_tensor.permute(0, 3, 4, 2, 1)
    # 展平空间维度
    flattened = temp_tensor.reshape(B, H * W * D, C)
    return flattened

def scan_front_to_back_inv(transformed_tensor, original_shape):
    """
    逆变换: 复原 scan_front_to_back 变换后的数据。
    输入形状: (B, D * H * W, C)
    复原形状: (B, C, D, H, W)
    """
    B, C, D, H, W = _get_original_dims(original_shape)
    # 恢复空间维度
    # (B, H*W*D, C) -> (B, H, W, D, C)
    recovered = transformed_tensor.view(B, H, W, D, C)
    # 恢复原始维度顺序
    # (B, H, W, D, C) -> (B, C, D, H, W)
    return recovered.permute(0, 4, 3, 1, 2)

def scan_back_to_front(input_tensor):
    """
    对输入张量进行沿深度 (D) 方向的从后到前扫描。
    输入形状: (B, C, D, H, W)
    变换后形状: (B, D * H * W, C)
    """
    B, C, D, H, W = _get_dims(input_tensor)
    # 首先沿深度维度进行翻转
    flipped_tensor = torch.flip(input_tensor, dims=[-3]) # 翻转 D 维度
    # 调整维度顺序，使 D 成为最内层空间维度，C 移到最后
    # (B, C, D, H, W) -> (B, H, W, D, C)
    temp_tensor = flipped_tensor.permute(0, 3, 4, 2, 1)
    # 展平空间维度
    flattened = temp_tensor.reshape(B, H * W * D, C)
    return flattened

def scan_back_to_front_inv(transformed_tensor, original_shape):
    """
    逆变换: 复原 scan_back_to_front 变换后的数据。
    输入形状: (B, D * H * W, C)
    复原形状: (B, C, D, H, W)
    """
    B, C, D, H, W = _get_original_dims(original_shape)
    # 恢复空间维度
    # (B, H*W*D, C) -> (B, H, W, D, C)
    recovered = transformed_tensor.view(B, H, W, D, C)
    # 恢复原始维度顺序
    # (B, H, W, D, C) -> (B, C, D, H, W)
    recovered = recovered.permute(0, 4, 3, 1, 2)
    # 最后沿深度维度进行翻转，抵消之前的翻转
    return torch.flip(recovered, dims=[-3])