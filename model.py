import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Base class for all models. Defines the common interface.
    """
    def __init__(self, config, dataset):
        super(Model, self).__init__()
        self.config = config
        self.dataset = dataset

    def forward(self):
        """
        Must be implemented in subclasses.
        """
        raise NotImplementedError

    def calculate_loss(self, users, items, labels):
        """
        Calculates the loss. Must be implemented in subclasses.
        """
        raise NotImplementedError

    def predict(self, users, items):
        """
        Predicts scores for given users and items.
        """
        raise NotImplementedError


class LightGCN(Model):
    """
    Implementation of LightGCN model using similarity_matrix.csv for edge weights.
    """
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        # Basic configurations
        self.n_users = dataset.n_users
        self.m_items = dataset.m_items
        self.latent_dim = config['latent_dim']
        self.n_layers = config['n_layers']

        # Embeddings for users and items
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.m_items, self.latent_dim)

        # Metadata features (optional)
        self.creator_features = dataset.get_creator_features()
        self.item_features = dataset.get_item_features()

        # Adjacency matrix from dataset
        self.adjacency = dataset.getSparseGraph()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initializes embeddings using Xavier uniform distribution.
        """
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self):
        """
        Forward pass for LightGCN.
        """
        # Initialize user and item embeddings
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        # Optionally integrate metadata features
        if self.creator_features:
            user_embeddings = self._integrate_metadata(user_embeddings, self.creator_features)
        if self.item_features:
            item_embeddings = self._integrate_metadata(item_embeddings, self.item_features)

        # Perform graph propagation
        all_embeddings = self.graph_propagation(user_embeddings, item_embeddings)

        return all_embeddings

    def _integrate_metadata(self, embeddings, features):
        """
        Integrates metadata features into node embeddings.
        """
        for key, feature in features.items():
            embeddings += feature
        return embeddings

    def graph_propagation(self, user_embeddings, item_embeddings):
        """
        Performs LightGCN graph propagation using the precomputed adjacency matrix.
        """
        # Combine user and item embeddings
        embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        all_embeddings = [embeddings]

        # Propagate through layers
        for layer in range(self.n_layers):
            embeddings = torch.sparse.mm(self.adjacency, embeddings)
            all_embeddings.append(embeddings)

        # Aggregate embeddings from all layers
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)

        # Split into user and item embeddings
        user_final, item_final = torch.split(all_embeddings, [self.n_users, self.m_items])
        return user_final, item_final

    def calculate_loss(self, users, items, labels):
        """
        Calculates BPR loss for the model.
        """
        user_embeddings, item_embeddings = self.forward()
        user_latent = user_embeddings[users]
        item_latent = item_embeddings[items]

        scores = torch.mul(user_latent, item_latent).sum(dim=1)
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        return loss

    def predict(self, users, items):
        """
        Predicts scores for given users and items.
        """
        user_embeddings, item_embeddings = self.forward()
        user_latent = user_embeddings[users]
        item_latent = item_embeddings[items]

        scores = torch.mul(user_latent, item_latent).sum(dim=1)
        return scores
