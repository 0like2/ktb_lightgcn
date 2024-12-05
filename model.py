import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def to_sparse_tensor(sparse_matrix):
    sparse_matrix = sparse_matrix.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_matrix.row, sparse_matrix.col))
    ).long()
    values = torch.from_numpy(sparse_matrix.data).float()
    shape = torch.Size(sparse_matrix.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

class Model(nn.Module):
    """
    Base class for all models. Defines the common interface.
    """
    def __init__(self, config, dataset):
        super(Model, self).__init__()
        self.config = config
        self.dataset = dataset

    def forward(self):
        raise NotImplementedError

    def calculate_loss(self, users, items, labels):
        raise NotImplementedError

    def predict(self, users, items):
        raise NotImplementedError


class LightGCN(Model):
    """
    Implementation of LightGCN model with metadata and similarity-based graph.
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

        # Metadata feature embeddings
        self.creator_features = dataset.get_creator_features()
        self.item_features = dataset.get_item_features()
        self.creator_feature_layers = self._create_feature_layers(self.creator_features)
        self.item_feature_layers = self._create_feature_layers(self.item_features)

        # Graph structure
        self.adjacency = to_sparse_tensor(dataset.getSparseGraph())

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def _create_feature_layers(self, features):
        """
        Creates a linear transformation layer for each feature.
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
        updated_embeddings = embeddings.clone()
        for key, feature in features.items():
            weight = self.config.get(f"{key}_weight", 1.0)
            updated_embeddings += layers[key](feature.float()) * weight
        return updated_embeddings

    def forward(self):
        """
        Forward pass for LightGCN with integrated metadata.
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        # Integrate metadata features into user and item embeddings
        user_embeddings = self._integrate_metadata(user_embeddings, self.creator_features, self.creator_feature_layers)
        item_embeddings = self._integrate_metadata(item_embeddings, self.item_features, self.item_feature_layers)

        # Perform graph propagation
        all_embeddings = self.graph_propagation(user_embeddings, item_embeddings)

        return all_embeddings

    def getUsersRating(self, users, creators_metadata, items_metadata, similarity_matrix):

        users_emb, items_emb = self.forward()
        users_emb = users_emb[users]
        scores = torch.matmul(users_emb, items_emb.T)

        user_categories = creators_metadata.iloc[users.cpu().numpy()]['channel_category'].values
        item_categories = items_metadata['item_category'].values
        batch_size = len(users)

        similarity_scores = torch.tensor(
            similarity_matrix[user_categories][:, item_categories],
            device=users_emb.device,
            dtype=torch.float32
        )

        scores += similarity_scores

        return scores

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

    def save_model(self, model_path, embedding_path):
        """
        Saves the model weights and embeddings.
        """
        torch.save(self.state_dict(), model_path)
        user_embeddings, item_embeddings = self.forward()
        torch.save({'user_embeddings': user_embeddings, 'item_embeddings': item_embeddings}, embedding_path)

    def load_model(self, model_path, embedding_path):
        """
        Loads the model weights and embeddings.
        """
        self.load_state_dict(torch.load(model_path))
        embeddings = torch.load(embedding_path)
        self.user_embedding.weight.data = embeddings['user_embeddings']
        self.item_embedding.weight.data = embeddings['item_embeddings']

    def evaluate(self, users, items, labels, top_k=10):
        """
        Evaluates the model using Precision, Recall, and NDCG.
        """
        with torch.no_grad():
            predictions = self.predict(users, items)
            top_k_indices = torch.topk(predictions, top_k).indices.cpu().numpy()

            precision, recall, ndcg = 0.0, 0.0, 0.0
            for i, user in enumerate(users):
                true_items = labels[i]
                recommended_items = top_k_indices[i]
                precision += len(set(recommended_items) & set(true_items)) / top_k
                recall += len(set(recommended_items) & set(true_items)) / len(true_items)
                dcg = sum(1 / np.log2(idx + 2) for idx, item in enumerate(recommended_items) if item in true_items)
                idcg = sum(1 / np.log2(idx + 2) for idx in range(min(len(true_items), top_k)))
                ndcg += dcg / idcg if idcg > 0 else 0

            precision /= len(users)
            recall /= len(users)
            ndcg /= len(users)

        return {'precision': precision, 'recall': recall, 'ndcg': ndcg}

    def predict(self, users, items):
        """
        Predicts scores for given users and items.
        """
        user_embeddings, item_embeddings = self.forward()
        user_latent = user_embeddings[users]
        item_latent = item_embeddings[items]

        scores = torch.sum(user_latent * item_latent, dim=1)
        return scores
