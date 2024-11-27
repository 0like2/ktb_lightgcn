import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, config, dataset, creator_features=None, item_features=None):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset

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

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def _create_feature_layers(self, features):
        """
        Creates feature transformation layers for metadata features.
        :param features: Dictionary of features (from dataset)
        :return: nn.ModuleDict of transformation layers
        """
        if features is None:
            return None

        layers = nn.ModuleDict()
        for feature_name, feature_data in features.items():
            input_dim = feature_data.size(-1)
            layers[feature_name] = nn.Linear(input_dim, self.latent_dim)
        return layers

    def forward(self):
        """
        Forward pass for LightGCN.
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        # Integrate metadata features if available
        if self.creator_features:
            creator_feature_embedding = self._process_features(
                self.creator_features, self.creator_feature_layers
            )
            user_embeddings = user_embeddings + creator_feature_embedding

        if self.item_features:
            item_feature_embedding = self._process_features(
                self.item_features, self.item_feature_layers
            )
            item_embeddings = item_embeddings + item_feature_embedding

        # Perform graph propagation
        all_embeddings = self.graph_propagation(user_embeddings, item_embeddings)

        return all_embeddings

    def _process_features(self, features, layers):
        """
        Processes metadata features through feature layers.
        :param features: Feature dictionary
        :param layers: Feature layers
        :return: Combined feature embedding
        """
        feature_embeddings = []
        for feature_name, feature_data in features.items():
            feature_embedding = layers[feature_name](feature_data)
            feature_embeddings.append(feature_embedding)
        return sum(feature_embeddings)

    def graph_propagation(self, user_embeddings, item_embeddings):
        """
        Performs LightGCN graph propagation.
        """
        embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        adjacency = self.dataset.getSparseGraph()

        all_embeddings = [embeddings]
        for layer in range(self.n_layers):
            embeddings = torch.sparse.mm(adjacency, embeddings)
            all_embeddings.append(embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
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
