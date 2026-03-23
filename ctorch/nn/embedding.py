from typing import TYPE_CHECKING, List

import torch
from .module import Module

class DeEmbedding(Module):
    def __init__(self, embedding: torch.nn.Embedding):
        '''
        De-embedding layer that maps from the embedding space back to the multinomial distribution over the vocabulary.

        Args:
            embedding (torch.nn.Embedding): The embedding layer to de-embed from.

        Shapes:

            * Input shape: (\\*, embedding.embedding_dim)
            * Output shape: (\\*, embedding.num_embeddings)
        '''
        super().__init__()
        self.embedding = embedding
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the de-embedding layer.

        Args:
            x (torch.Tensor): Tensor of shape (\\*, D), where D is the embedding dimension.

        Returns:
            torch.Tensor: Tensor of shape (\\*, num_embeddings), where num_embeddings is the size of the embedding.
        '''
        if x.dim() < 2:
            raise ValueError('Input tensor must have at least 2 dimensions.')
        if x.shape[-1] != self.embedding.embedding_dim:
            raise ValueError(f'Input tensor must have last dimension of size {self.embedding.embedding_dim}, but got {x.shape[-1]}.')

        ret = torch.einsum('...d,ed->...e', x, self.embedding.weight)
        return self.softmax(ret)

class FeatureEmbedding(Module):
    '''
    An embedding layer for encoding N multiple categorical features.

    Args:
        num_features (List[int]): The number of unique values for each categorical feature.
        embedding_size (List[int] | int): The size of the embedding for each feature. If a single integer is provided, it will be used for all features.
        padding_idx (int | None):
        max_norm (float | None):
        norm_type (float):
        scale_grad_by_freq (bool):
        sparse (bool):

    Shapes:

        * Input shape: (\\*, num_features)
        * Output shape: (\\*, sum(embedding_size))
    '''
    def __init__(
        self, num_features: List[int], embedding_size: List[int] | int,
        padding_idx: int | None = None, max_norm: float | None = None,
        norm_type: float = 2.0, scale_grad_by_freq: bool = False,
        sparse: bool = False
    ):
        super().__init__()
        # Type conversion and sanity checks
        if isinstance(embedding_size, int):
            embedding_size = [embedding_size] * len(num_features)
        if len(num_features) != len(embedding_size):
            raise ValueError('num_features and embedding_size must have the same length.')

        # Normalization should be done separately
        self.num_features = num_features
        num_features_tensor = torch.tensor(num_features, dtype=torch.long).unsqueeze(0)
        if TYPE_CHECKING:
            self.num_features_tensor = num_features_tensor
        self.register_buffer('num_features_tensor', num_features_tensor)
        self.embedding_size = embedding_size

        _total_embeddings = sum(num_features + [0])
        _max_embedding_size = max(embedding_size + [0])
        self.embedding = torch.nn.Embedding(
            num_embeddings=_total_embeddings,
            embedding_dim=_max_embedding_size,
            padding_idx=padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
            max_norm=max_norm,
            norm_type=norm_type,
            sparse=sparse
        )

        # If any embedding size is different from the max, we need to slice the output
        self.need_slice = any(_ != _max_embedding_size for _ in embedding_size)

        # Offset is used to calculate the start index of each feature's embedding
        offset = torch.cumsum(torch.tensor([0, *self.num_features[:-1]]), dim=0, dtype=torch.long)
        if TYPE_CHECKING:
            self.offset = offset
        self.register_buffer('offset', offset)

    @property
    def total_embedding_size(self) -> int:
        '''
        Gets the total embedding size, which is the sum of all individual embedding sizes.

        Returns:
            int: Total embedding size.
        '''
        return sum(self.embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the embedding layer.

        Args:
            x: Tensor of shape (\\*, num_features)

        Returns:
            torch.Tensor: Tensor of shape (\\*, sum(embedding_size))
        '''
        # Sanity checks
        if x.dim() < 2:
            raise ValueError('Input tensor must have at least 2 dimensions.')
        if x.shape[-1] != len(self.num_features):
            raise ValueError(f'Input tensor must have last dimension of size {len(self.num_features)}, but got {x.shape[-1]}.')
        if x.dtype not in (torch.int, torch.long):
            raise TypeError('Input tensor must be of type torch.int/long.')
        if not torch.all((x >= 0).all(dim=-1) & (x < self.num_features_tensor).all(dim=-1)):
            raise ValueError('Input tensor contains out-of-bound indices.')

        # Flatten the last dim
        *shape, num_features = x.shape
        x = x.view(*shape, -1)
        x = x + self.offset.expand(*shape, -1)  # Add offset to each feature's index

        # Get embeddings
        embeddings = self.embedding(x)

        # Return sliced embeddings if needed
        if self.need_slice:
            embeddings = torch.cat([
                embeddings[..., i:i + size]
                for i, size in zip(self.offset, self.embedding_size)
            ], dim=-1)

        return embeddings.view(*shape, -1)
