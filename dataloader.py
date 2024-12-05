import pandas as pd
from scipy.sparse import csr_matrix, vstack, hstack
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.nn import Embedding
from sentence_transformers import SentenceTransformer
from model import to_sparse_tensor
from world import cprint


class BasicDataset:
    def __init__(self):
        raise NotImplementedError("This is an interface class. Do not instantiate directly.")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def similarity_matrix(self):
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
        if not text or text.strip() == "":  # Handle empty or None strings
            return np.zeros(384)  # Default embedding size
        try:
            return self.model.encode(text)
        except Exception as e:
            print(f"Error encoding text '{text}': {e}")
            return np.zeros(384)  # Return zero vector on failure


class SimilarityDataset(BasicDataset):
    """
    Dataset class for category similarity-based LightGCN.
    """
    def __init__(self, creator_file, item_file, similarity_matrix_file, config, threshold=0.05):
        """
        Initializes the dataset using metadata and precomputed similarity matrix.
        """
        self.config=config
        self.similarity_matrix_file = similarity_matrix_file
        cprint("Loading metadata and similarity matrix...")
        self.creators = pd.read_csv(creator_file)
        self.items = pd.read_csv(item_file)
        try:
            self._similarity_matrix = pd.read_csv(similarity_matrix_file, index_col=0).values  # 수정: 내부 변수 사용
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {similarity_matrix_file}")
        except Exception as e:
            raise ValueError(f"Error reading similarity matrix file: {e}")

        # Metadata preprocessing
        self.scaler = MinMaxScaler()
        self.text_embedder = TextEmbedder()
        self.category_embedding_layer = Embedding(num_embeddings=100, embedding_dim=16)

        # Validate and preprocess data
        self.creators = self._validate_creators(self.creators)
        self.items = self._validate_items(self.items)

        # Metadata feature extraction
        self.creators['normalized_subscribers'] = self.normalize_subscribers(
            self.creators['subscribers'], max_value=10000000, scale=100
        )
        self.creators['name_embedding'] = self.creators['channel_name'].apply(self.text_embedder.get_text_embedding)
        self.items['title_embedding'] = self.items['title'].apply(self.text_embedder.get_text_embedding)
        self.items['content_embedding'] = self.items['item_content'].apply(self.text_embedder.get_text_embedding)

        # Graph construction
        self.graph = self._build_graph()

        # Test data generation
        self.testUser = np.random.choice(self.n_users, size=min(self.n_users, 10), replace=False)  # Sample test users
        self.__testDict = self.__build_test(threshold=threshold)

    @property
    def similarity_matrix(self):
        return self._similarity_matrix

    def _validate_creators(self, creators):
        """
        Validates and preprocesses creator metadata.
        """
        creators['channel_category'] = creators['channel_category'].fillna('unknown').astype("category").cat.codes
        creators['channel_name'] = creators['channel_name'].fillna('unknown')
        creators['subscribers'] = creators['subscribers'].replace({',': ''}, regex=True).fillna(0).astype(float)
        return creators

    def _validate_items(self, items):
        """
        Validates and preprocesses item metadata.
        """
        items['title'] = items['title'].fillna('unknown')
        items['item_category'] = items['item_category'].fillna('unknown').astype("category").cat.codes
        items['media_type'] = items['media_type'].fillna('unknown').map({'short': 0, 'long': 1}).fillna(0).astype(int)
        items['score'] = items['score'].fillna(0).astype(int)
        items['item_content'] = items['item_content'].fillna('unknown')
        return items

    def normalize_subscribers(self, subscribers, max_value, scale=100):
        """
        Normalizes the subscriber count to a scale of 0 to 100.
        """
        normalized = np.round((subscribers / max_value) * scale).astype(int)
        return np.clip(normalized, 0, scale)

    from scipy.sparse import csr_matrix, vstack, hstack

    def _build_graph(self):
        # 디버깅 -> 삭제
        print("[DEBUG] Building graph...")
        print(f"[DEBUG] similarity_matrix shape: {self.similarity_matrix.shape}")
        print(f"[DEBUG] n_users: {self.n_users}, m_items: {self.m_items}")
        print(f"[DEBUG] creators shape: {len(self.creators)}, items shape: {len(self.items)}")

        n_users = self.n_users
        m_items = self.m_items

        # 사용자-아이템 관계 그래프 생성
        user_item_graph = self._build_user_item_graph()

        # 사용자-사용자 연결 (빈 행렬)
        user_user_graph = csr_matrix((n_users, n_users))
        # 아이템-아이템 연결 (빈 행렬)

        item_item_graph = csr_matrix((m_items, m_items))

        # 상단 행렬 (사용자-사용자 | 사용자-아이템)
        top = hstack([user_user_graph, user_item_graph])
        # 하단 행렬 (아이템-사용자 | 아이템-아이템)
        bottom = hstack([user_item_graph.T, item_item_graph])

        # 전체 그래프 결합
        graph = vstack([top, bottom])

        print(f"[DEBUG] Final Graph Shape: {graph.shape}")
        print(f"[DEBUG] Non-zero Entries (Final Graph): {graph.nnz}")

        return graph

    def __build_test(self, threshold=0.85):
        """
        Builds the test dictionary based on user and item category similarity.
        Ensures test items do not overlap with training items.
        """
        test_data = {}

        for user_id in range(self.n_users):
            user_category = self.creators.iloc[user_id]['channel_category']  # 유저 카테고리 가져오기
            pos_items = []

            for item_id in range(self.m_items):
                item_category = self.items.iloc[item_id]['item_category']  # 아이템 카테고리 가져오기
                similarity_score = self.similarity_matrix[user_category, item_category]

                # 유사도가 threshold 이상인 경우 테스트 데이터로 추가
                if similarity_score >= threshold:
                    pos_items.append(item_id)

            # 학습 데이터(allPos)와 겹치지 않도록 필터링
            train_items = set(self.allPos[user_id])
            test_items = [item for item in pos_items if item not in train_items]

            test_data[user_id] = test_items

            # 디버깅 출력
            print(f"[DEBUG] User ID: {user_id}, Test Items: {test_items}")

        return test_data

    def _build_user_item_graph(self):
        """
        Builds the user-item graph based on the similarity matrix and user-item metadata.
        """
        n_users = self.n_users
        m_items = self.m_items

        # 사용자-아이템 관계 그래프 생성
        user_item_graph = np.zeros((n_users, m_items))

        for user_id, user_data in enumerate(self.creators.itertuples()):
            # user_id로부터 user_category 가져오기
            user_category = user_data.channel_category

            for item_id, item_data in enumerate(self.items.itertuples()):
                # item_id로부터 item_category 가져오기
                item_category = item_data.item_category

                # similarity_matrix에서 user_category와 item_category 간 유사도 가져오기
                similarity_score = self.similarity_matrix[user_category, item_category]

                # 유사도가 0보다 큰 경우 그래프에 추가
                if similarity_score > 0:
                    user_item_graph[user_id, item_id] = similarity_score

        return csr_matrix(user_item_graph)



    def getSparseGraph(self):
        """
        Returns the precomputed sparse graph.
        """
        return self.graph

    def get_creator_features(self):
        """
        Returns features for creators.
        """
        return {
            'category_embedding': self.category_embedding_layer(
                torch.tensor(self.creators['channel_category'].values, dtype=torch.long)
            ),
            'name_embedding': torch.tensor(np.vstack(self.creators['name_embedding'].values)),
            'normalized_subscribers': torch.tensor(self.creators['normalized_subscribers']).unsqueeze(-1),
        }

    def get_item_features(self):
        """
        Returns features for items.
        """
        return {
            'category_embedding': self.category_embedding_layer(
                torch.tensor(self.items['item_category'].values, dtype=torch.long)
            ),
            'title_embedding': torch.tensor(np.vstack(self.items['title_embedding'].values)),
            'media_type': torch.tensor(self.items['media_type']),
            'score': torch.tensor(self.items['score'].values).unsqueeze(-1),
            'content_embedding': torch.tensor(np.vstack(self.items['content_embedding'].values)),
        }

    def get_creator_item_data(self):
        """
        Returns raw creator and item data.
        """
        return self.creators, self.items

    @property
    def n_users(self):
        return len(self.creators)

    @property
    def m_items(self):
        return len(self.items)

    @property
    def trainDataSize(self):
        return self.graph.nnz

    @property
    def allPos(self):
        """
        Generates positive interactions based on user-item graph.
        """
        all_pos = {user: [] for user in range(self.n_users)}
        user_item_graph = self._build_user_item_graph()

        # 사용자-아이템 관계를 `allPos`로 변환
        rows, cols = user_item_graph.nonzero()
        for user, item in zip(rows, cols):
            all_pos[user].append(item)

        print(f"[DEBUG] All Positive Interactions: {len(all_pos)} users.")
        return all_pos

    @property
    def allNeg(self):
        """
        Generates negative interactions based on similarity threshold.
        Only includes items with similarity below 0.35.
        """
        all_neg = {user: [] for user in range(self.n_users)}

        for user_id in range(self.n_users):
            user_category = self.creators.iloc[user_id]['channel_category']  # 유저 카테고리 가져오기
            neg_items = []

            for item_id in range(self.m_items):
                item_category = self.items.iloc[item_id]['item_category']  # 아이템 카테고리 가져오기

                # similarity_matrix에서 유저-아이템 간 유사도 가져오기
                similarity_score = self.similarity_matrix[user_category, item_category]

                # 유사도가 0.35 이하인 경우 부정적 샘플로 추가
                if similarity_score <= 0.35:
                    neg_items.append(item_id)

            all_neg[user_id] = neg_items

        print(f"[DEBUG] All Negative Interactions: {len(all_neg)} users.")
        return all_neg

    @property
    def testDict(self):
        return self.__testDict
