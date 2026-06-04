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

    def output_shape(
        self, *args: torch.Size | None, **kwargs: torch.Size | None
    ) -> torch.Size | None:
        x_shape = args[0]
        if x_shape is None:
            return None
        return torch.Size([*x_shape[:-1], self.embedding.num_embeddings])

    def guard_input_shape(self, *args, **kwargs):
        x = args[0]
        expected = self.embedding.embedding_dim
        if x.shape[-1] != expected:
            raise ValueError(
                f'{self.__class__.__name__}: expected input dim {expected}, '
                f'got {x.shape[-1]}'
            )

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
        if self._debug:
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
                embeddings[..., i, :size]
                for i, size in enumerate(self.embedding_size)
            ], dim=-1)

        return embeddings.view(*shape, -1)

    def output_shape(
        self, *args: torch.Size | None, **kwargs: torch.Size | None
    ) -> torch.Size | None:
        x_shape = args[0]
        if x_shape is None:
            return None
        return torch.Size([*x_shape[:-1], self.total_embedding_size])

    def guard_input_shape(self, *args, **kwargs):
        x = args[0]
        expected = len(self.num_features)
        if x.shape[-1] != expected:
            raise ValueError(
                f'{self.__class__.__name__}: expected {expected} features, '
                f'got {x.shape[-1]}'
            )

class SelfAttentionEmbedding(Module):
    '''
    An embedding layer for encoding N multiple categorical features. The module uses self-attention to perform cross-feature interaction.

    Args:
        num_features (List[int]): The number of unique values for each categorical feature.
        embedding_size (int): The size of the embedding for each feature.
        padding_idx (int | None): The global padding index.
        max_norm (float | None):
        norm_type (float):
        scale_grad_by_freq (bool):
        sparse (bool):

    Shapes:

        * Input shape: (\\*, num_features)
        * Output shape: (\\*, embedding_size * len(num_features))
    '''
    def __init__(
        self, num_features: List[int], embedding_size: int,
        padding_idx: int | None = None, max_norm: float | None = None,
        norm_type: float = 2.0, scale_grad_by_freq: bool = False,
        sparse: bool = False
    ):
        super().__init__()
        # Sanity checks
        if embedding_size % 2 or embedding_size <= 0:
            raise ValueError(f'Invalid embedding_size {embedding_size}.')

        # Normalization should be done separately
        self.num_features = num_features.copy()
        num_features_tensor = torch.tensor(num_features, dtype=torch.long).unsqueeze(0)
        if TYPE_CHECKING:
            self.num_features_tensor = num_features_tensor
        self.register_buffer('num_features_tensor', num_features_tensor)
        self.embedding_size = embedding_size

        _total_embeddings = sum(num_features + [0])
        self.embedding = torch.nn.Embedding(
            num_embeddings=_total_embeddings,
            embedding_dim=embedding_size,
            padding_idx=padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
            max_norm=max_norm,
            norm_type=norm_type,
            sparse=sparse
        )

        # Offset is used to calculate the start index of each feature's embedding
        offset = torch.cumsum(torch.tensor([0, *self.num_features[:-1]]), dim=0, dtype=torch.long)
        if TYPE_CHECKING:
            self.offset = offset
            self.sqrt: torch.Tensor
        self.register_buffer('offset', offset)
        self.register_buffer('sqrt', torch.tensor((embedding_size // 2) ** 0.5))

    @property
    def total_embedding_size(self) -> int:
        '''
        Gets the total embedding size, which is the sum of all individual embedding sizes.

        Returns:
            int: Total embedding size.
        '''
        return len(self.num_features) * self.embedding_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the embedding layer.

        Args:
            x: Tensor of shape (\\*, num_features)

        Returns:
            torch.Tensor: Tensor of shape (\\*, sum(embedding_size))
        '''
        # Sanity checks
        if self._debug:
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
        x = x + self.offset.expand(*shape, -1)  # Add offset to each feature's index

        # Get embeddings
        embeddings = self.embedding(x) # shape: (\\*, num_features, embedding_size)

        qk, v = embeddings.chunk(2, dim=-1)
        sa_weights = torch.softmax(
            torch.einsum('...qe,...ke->...qk', qk, qk) / self.sqrt,
            dim=-1
        )
        sa_embedding = torch.einsum('...qk,...ke->...qe', sa_weights, v)
        return torch.cat([qk, sa_embedding], dim=-1).reshape(*shape, -1)

    def output_shape(
        self, *args: torch.Size | None, **kwargs: torch.Size | None
    ) -> torch.Size | None:
        x_shape = args[0]
        if x_shape is None:
            return None
        return torch.Size([*x_shape[:-1], self.total_embedding_size])

    def guard_input_shape(self, *args, **kwargs):
        x = args[0]
        expected = len(self.num_features)
        if x.shape[-1] != expected:
            raise ValueError(
                f'{self.__class__.__name__}: expected {expected} features, '
                f'got {x.shape[-1]}'
            )
