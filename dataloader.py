import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.nn import Embedding
from sentence_transformers import SentenceTransformer
from world import cprint


class BasicDataset:
    """
    Base class for datasets in LightGCN.
    Defines the interface for all dataset types.
    """
    def __init__(self):
        raise NotImplementedError("This is an interface class. Do not instantiate directly.")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getSparseGraph(self):
        raise NotImplementedError


class TextEmbedder:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def get_text_embedding(self, text):
        if not text or text.strip() == "":  # 빈 문자열 또는 None 체크
            return np.zeros(384)  # 모델 임베딩 크기에 맞게 0 벡터 반환
        try:
            return self.model.encode(text)
        except Exception as e:
            print(f"Error encoding text '{text}': {e}")
            return np.zeros(384)  # 에러 발생 시 0 벡터 반환


class SimilarityDataset:
    """
    Dataset class for category similarity-based LightGCN with individual metadata features.
    """
    def __init__(self, creator_file, item_file, similarity_matrix_file, threshold=0.5):
        """
        Initializes the dataset using metadata and precomputed similarity matrix.
        :param creator_file: Path to the creator metadata CSV file.
        :param item_file: Path to the item metadata CSV file.
        :param similarity_matrix_file: Path to the precomputed similarity matrix CSV file.
        :param threshold: Similarity threshold for graph construction.
        """
        self.similarity_matrix_file = None
        cprint("Loading metadata and similarity matrix")
        self.creators = pd.read_csv(creator_file)
        self.items = pd.read_csv(item_file)
        self.similarity_matrix = pd.read_csv(similarity_matrix_file, index_col=0)
        self.threshold = threshold

        # Metadata preprocessing
        self.scaler = MinMaxScaler()
        self.creators['normalized_subscribers'] = self.normalize_subscribers(self.creators['subscribers'])
        self.text_embedder = TextEmbedder()
        self.category_embedding_layer = Embedding(num_embeddings=100, embedding_dim=16)

        # Text embedding
        self.items['title_embedding'] = self.items['title'].apply(self.text_embedder.get_text_embedding)
        self.creators['name_embedding'] = self.creators['channel_name'].apply(self.text_embedder.get_text_embedding)

        # Create sparse graph
        self.graph = self._build_graph()

    def normalize_subscribers(self, subscribers):
        """
        Normalizes the subscriber count using Min-Max scaling.
        """
        subscribers = subscribers.replace(',', '', regex=True).astype(float)
        return self.scaler.fit_transform(subscribers.values.reshape(-1, 1)).flatten()

    def load_similarity_matrix(self):
        """
        Loads the precomputed similarity matrix from a CSV file.
        """
        return pd.read_csv(self.similarity_matrix_file, index_col=0)

    def calculate_category_similarity(self, category_1, category_2):
        """
        Fetches the similarity value between two categories from the similarity matrix.
        """
        if category_1 in self.similarity_matrix.columns and category_2 in self.similarity_matrix.columns:
            return self.similarity_matrix.loc[category_1, category_2]
        return 0.5  # Default similarity if category is not found

    def _build_graph(self):
        """
        Builds a sparse graph from the similarity matrix.
        """
        similarity_values = self.similarity_matrix.values
        rows, cols = np.where(similarity_values > self.threshold)
        data = similarity_values[rows, cols]
        graph = csr_matrix((data, (rows, cols)), shape=similarity_values.shape)
        return graph

    def getSparseGraph(self):
        """
        Returns the sparse graph.
        """
        return self.graph

    def get_creator_features(self):
        """
        Returns independent metadata features for creators.
        """
        features = {
            'category_embedding': self.category_embedding_layer(
                torch.tensor(pd.factorize(self.creators['creator_category'])[0])
            ),
            'name_embedding': torch.tensor(np.vstack(self.creators['name_embedding'].values)),
            'normalized_subscribers': torch.tensor(self.creators['normalized_subscribers']).unsqueeze(-1),
        }
        return features

    def get_item_features(self):
        """
        Returns independent metadata features for items.
        """
        features = {
            'category_embedding': self.category_embedding_layer(
                torch.tensor(pd.factorize(self.items['item_category'])[0])
            ),
            'title_embedding': torch.tensor(np.vstack(self.items['title_embedding'].values)),
            'media_type': torch.tensor(
                pd.factorize(self.items['media_type'])[0]
            ),  # Categorical feature
            'score': torch.tensor(self.items['score'].values).unsqueeze(-1),
        }
        return features

    def get_creator_item_data(self):
        """
        Returns the creator and item data.
        """
        return self.creators, self.items

    @property
    def n_users(self):
        """
        Number of creators.
        """
        return len(self.creators)

    @property
    def m_items(self):
        """
        Number of items.
        """
        return len(self.items)

    @property
    def trainDataSize(self):
        """
        Size of training data (based on graph edges).
        """
        return self.graph.nnz

    @property
    def testDict(self):
        """
        Test data dictionary (to be defined if needed).
        """
        raise NotImplementedError("Test data is not implemented for this dataset.")

    @property
    def allPos(self):
        """
        All positive interactions (edges in the graph).
        """
        return self.graph.nonzero()

