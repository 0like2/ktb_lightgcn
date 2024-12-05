import torch
from model import LightGCN
from dataloader import SimilarityDataset
import world


class LightGCNRecommender:
    def __init__(self, model_path, dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load LightGCN model
        self.model = LightGCN(dataset.config, dataset).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Load metadata
        self.dataset = dataset
        self.user_embeddings, self.item_embeddings = self.model.get_embeddings()

    def preprocess_new_item(self, item_data):
        """
        Preprocesses new item data for recommendation.
        """
        item_category = self.dataset.similarity_matrix.columns.tolist().index(
            item_data.get('item_category', 'unknown')
        )
        media_type = 0 if item_data.get('media_type', '').lower() == 'short' else 1

        item_embedding = torch.tensor(
            self.dataset.text_embedder.get_text_embedding(item_data.get('title', 'unknown')),
            dtype=torch.float
        ).to(self.device)

        processed_item = {
            'item_id': self.dataset.n_items - 1,  # Use the next available item ID
            'item_category': item_category,
            'media_type': media_type,
            'item_embedding': item_embedding,
            'subscribers': 0,  # Items don't have subscribers
            'channel_category': 0,  # Default channel category for items
            'creator_embedding': torch.zeros(768)  # Default creator embedding
        }
        return processed_item

    def preprocess_new_creator(self, creator_data):
        """
        Preprocesses new creator data for recommendation.
        """
        channel_category = self.dataset.similarity_matrix.columns.tolist().index(
            creator_data.get('channel_category', 'unknown')
        )

        normalized_subscribers = self.dataset.normalize_subscribers(
            [creator_data.get('subscribers', 0)]
        )[0]

        creator_embedding = torch.tensor(
            self.dataset.text_embedder.get_text_embedding(creator_data.get('channel_name', 'unknown')),
            dtype=torch.float
        ).to(self.device)

        processed_creator = {
            'creator_id': self.dataset.n_users - 1,  # Use the next available creator ID
            'channel_category': channel_category,
            'creator_embedding': creator_embedding,
            'subscribers': normalized_subscribers,
            'item_category': 0,  # Default item category for creators
            'media_type': 0,  # Default media type for creators
            'item_embedding': torch.zeros(384)  # Default item embedding
        }
        return processed_creator

    def recommend_for_new_item(self, item_data, top_k=10):
        """
        Recommends users for a new item.
        """
        processed_item = self.preprocess_new_item(item_data)

        # Compute similarity between the new item and all user embeddings
        scores = torch.matmul(self.user_embeddings, processed_item['item_embedding'])
        top_k_indices = torch.topk(scores, top_k).indices.cpu().numpy()

        # Retrieve recommended user metadata
        recommended_users = [
            {
                'creator_id': int(i),
                'channel_name': self.dataset.creators.iloc[i]['channel_name'],
                'channel_category': self.dataset.creators.iloc[i]['creator_category'],
                'subscribers': int(self.dataset.creators.iloc[i]['subscribers']),
            }
            for i in top_k_indices
        ]
        return recommended_users

    def recommend_for_new_creator(self, creator_data, top_k=10):
        """
        Recommends items for a new creator.
        """
        processed_creator = self.preprocess_new_creator(creator_data)

        # Compute similarity between the new creator and all item embeddings
        scores = torch.matmul(self.item_embeddings, processed_creator['creator_embedding'])
        top_k_indices = torch.topk(scores, top_k).indices.cpu().numpy()

        # Retrieve recommended item metadata
        recommended_items = [
            {
                'item_id': int(i),
                'title': self.dataset.items.iloc[i]['title'],
                'item_category': self.dataset.items.iloc[i]['item_category'],
                'media_type': self.dataset.items.iloc[i]['media_type'],
                'item_score': self.dataset.items.iloc[i]['score'],
                'item_content': self.dataset.items.iloc[i]['item_content'],  # Add 'item_content' field
            }
            for i in top_k_indices
        ]
        return recommended_items


if __name__ == "__main__":
    # Paths to model and dataset
    model_path = "output/lightgcn_model.pth"

    # Initialize dataset and recommender
    dataset = SimilarityDataset(
        creator_file="path/to/creator_file.csv",
        item_file="path/to/item_file.csv",
        similarity_matrix_file="path/to/similarity_matrix.csv",
        config=world.config  # 수정: config 추가
    )
    recommender = LightGCNRecommender(model_path, dataset)

    # New item example
    new_item_data = {
        'title': "바밤바를 뛰어넘는 밤 맛 과자가 있을까?",
        'item_category': 'entertainment',
        'media_type': 'short',
        'score': 80,
        'item_content': '다양한 밤 맛 과자를 비교하며 맛과 질감을 리뷰하는 콘텐츠'
    }

    # Recommend users for the new item
    recommended_users = recommender.recommend_for_new_item(new_item_data)
    print(f"추천 사용자 목록: {recommended_users}")

    # New creator example
    new_creator_data = {
        'channel_category': "tech",
        'channel_name': "최마태의 POST IT",
        'subscribers': 263000
    }

    # Recommend items for the new creator
    recommended_items = recommender.recommend_for_new_creator(new_creator_data)
    print(f"추천 아이템 목록: {recommended_items}")
