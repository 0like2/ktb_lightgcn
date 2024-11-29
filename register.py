import world
import dataloader
import model
import utils
from pprint import pprint

# Dataset initialization
if world.dataset == 'lastfm':
    # For debugging purposes
    print("Loading LastFM dataset for debugging...")
    dataset = dataloader.LastFM()
elif world.dataset == 'custom-similarity':
    # For custom similarity-based dataset
    dataset = dataloader.SimilarityDataset(
        creator_file=world.config['creator_file'],
        item_file=world.config['item_file'],
        similarity_matrix_file=world.config['similarity_matrix_file'],
        threshold=world.config.get('threshold', 0.5)
    )
else:
    raise ValueError(f"Unknown dataset: {world.dataset}")

# Print configuration details
print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

# Model selection
MODELS = {
    'mf': model.PureMF,
    'lgn': lambda config, dataset: model.LightGCN(
        config=config,
        dataset=dataset,
        creator_features=dataset.get_creator_features() if hasattr(dataset, 'get_creator_features') else None,
        item_features=dataset.get_item_features() if hasattr(dataset, 'get_item_features') else None
    )
}

# Model initialization
if world.model_name not in MODELS:
    raise ValueError(f"Unknown model: {world.model_name}")

Recmodel = MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
