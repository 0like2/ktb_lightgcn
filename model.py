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
        if features is None:
            return None

        layers = nn.ModuleDict()
        for feature_name, feature_data in features.items():
            # Determine input dimension based on feature shape
            input_dim = feature_data.size(-1) if len(feature_data.size()) > 1 else 1
            layers[feature_name] = nn.Linear(input_dim, self.latent_dim)
        return layers

    def _integrate_metadata(self, embeddings, features, layers):
        if features is None or layers is None:
            return embeddings

        transformed_features = []
        for key, feature in features.items():
            # Debugging: Print feature shapes

            if len(feature.size()) == 1:
                feature = feature.unsqueeze(-1)  # Reshape (batch_size, 1)

            weight = self.config.get(f"{key}_weight", 1.0)
            transformed = layers[key](feature.float()) * weight
            transformed_features.append(transformed)

        transformed_features = torch.sum(torch.stack(transformed_features), dim=0)
        updated_embeddings = embeddings + transformed_features
        return updated_embeddings

    def forward(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        user_embeddings = self._integrate_metadata(user_embeddings, self.creator_features, self.creator_feature_layers)
        item_embeddings = self._integrate_metadata(item_embeddings, self.item_features, self.item_feature_layers)
        all_embeddings = self.graph_propagation(user_embeddings, item_embeddings)
        return all_embeddings

    def getUsersRating(self, users, creators_metadata_tensor, items_metadata_tensor, similarity_matrix):
        # Forward pass to get embeddings
        users_emb, items_emb = self.forward()
        users_emb = users_emb[users]
        scores = torch.matmul(users_emb, items_emb.T)

        # Ensure indices are in long type for tensor indexing
        user_categories = creators_metadata_tensor[users, 0].long()
        item_categories = items_metadata_tensor[:, 0].long()

        # Similarity score 계산
        similarity_scores = similarity_matrix[user_categories][:, item_categories]
        similarity_scores = similarity_scores.clone().detach().to(users_emb.device)

        scores += similarity_scores
        return scores

    def graph_propagation(self, user_embeddings, item_embeddings):
        embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        all_embeddings = [embeddings]

        layer_weights = nn.Parameter(torch.ones(self.n_layers + 1))

        for layer in range(self.n_layers):
            embeddings = torch.sparse.mm(self.adjacency, embeddings)
            all_embeddings.append(embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        weighted_embeddings = all_embeddings * layer_weights.view(1, -1, 1)
        final_embeddings = weighted_embeddings.sum(dim=1)

        user_final, item_final = torch.split(final_embeddings, [self.n_users, self.m_items])
        return user_final, item_final

    def calculate_loss(self, users, pos_items, neg_items):
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

        # Debugging: Print loss values
        print(f"[DEBUG] Loss: {loss.item()}, Regularization Loss: {reg_loss.item()}")

        return loss, reg_loss

    def bpr_loss(self, users, pos_items, neg_items):
        """
        Wrapper for BPR loss calculation using calculate_loss.
        """
        return self.calculate_loss(users, pos_items, neg_items)

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
