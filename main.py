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
    Implementation of LightGCN model with metadata and similarity-based graph.
    """
    def __init__(self, config, dataset, creator_features=None, item_features=None):
        super(LightGCN, self).__init__(config, dataset)

        # Basic configurations
        self.n_users = dataset.n_users
        self.m_items = dataset.m_items
        self.latent_dim = config['latent_dim']
        self.n_layers = config['n_layers']

        # Embeddings for users and items
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.m_items, self.latent_dim)

        # Metadata feature embeddings (optional)
        self.creator_features = creator_features
        self.item_features = item_features
        self.creator_feature_layers = self._create_feature_layers(creator_features)
        self.item_feature_layers = self._create_feature_layers(item_features)

        # Graph structure
        self.adjacency = dataset.getSparseGraph()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def _create_feature_layers(self, features):
        """
        Creates feature transformation layers for metadata features.
        """
        if features is None:
            return None

        layers = nn.ModuleDict()
        for feature_name, feature_data in features.items():
            input_dim = feature_data.size(-1)
            layers[feature_name] = nn.Linear(input_dim, self.latent_dim)
        return layers

    def _integrate_metadata(self, embeddings, features, layers):
        """
        Integrates metadata features into embeddings.
        """
        if features is None or layers is None:
            return embeddings
        for key, feature in features.items():
            embeddings += layers[key](feature)
        return embeddings

    def forward(self):
        """
        Forward pass for LightGCN.
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        # Integrate metadata features if available
        user_embeddings = self._integrate_metadata(user_embeddings, self.creator_features, self.creator_feature_layers)
        item_embeddings = self._integrate_metadata(item_embeddings, self.item_features, self.item_feature_layers)

        # Perform graph propagation
        all_embeddings = self.graph_propagation(user_embeddings, item_embeddings)

        return all_embeddings

    def graph_propagation(self, user_embeddings, item_embeddings):
        """
        Performs LightGCN graph propagation.
        """
        embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)

        all_embeddings = [embeddings]
        for layer in range(self.n_layers):
            embeddings = torch.sparse.mm(self.adjacency, embeddings)
            all_embeddings.append(embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_final, item_final = torch.split(all_embeddings, [self.n_users, self.m_items])
        return user_final, item_final

    def calculate_loss(self, users, pos_items, neg_items):
        """
        Calculates BPR loss for the model.
        """
        user_embeddings, item_embeddings = self.forward()
        user_latent = user_embeddings[users]
        pos_latent = item_embeddings[pos_items]
        neg_latent = item_embeddings[neg_items]

        pos_scores = torch.sum(user_latent * pos_latent, dim=-1)
        neg_scores = torch.sum(user_latent * neg_latent, dim=-1)

        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        reg_loss = (user_latent.norm(2).pow(2) +
                    pos_latent.norm(2).pow(2) +
                    neg_latent.norm(2).pow(2)) / 2
        return loss + self.config['decay'] * reg_loss

    def getUsersRating(self, users):
        """
        Predicts scores for all items for given users.
        """
        user_embeddings, item_embeddings = self.forward()
        user_latent = user_embeddings[users]
        scores = torch.matmul(user_latent, item_embeddings.T)
        return scores

    def predict(self, users, items):
        """
        Predicts scores for given users and items.
        """
        user_embeddings, item_embeddings = self.forward()
        user_latent = user_embeddings[users]
        item_latent = item_embeddings[items]

        scores = torch.sum(user_latent * item_latent, dim=1)
        return scores
