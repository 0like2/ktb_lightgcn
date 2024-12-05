import world
import dataloader
import model
from pprint import pprint

# Dataset initialization
if world.dataset == 'custom-similarity':
    dataset = dataloader.SimilarityDataset(
        creator_file=world.config['creator_file'],
        item_file=world.config['item_file'],
        similarity_matrix_file=world.config['similarity_matrix_file'],
        config=world.config
    )
    print(f"Dataset initialized: {type(dataset)} with {len(dataset.creators)} creators and {len(dataset.items)} items.") # 디버깅 -> 삭제
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
    'lgn': lambda config, dataset: model.LightGCN(
        config=config,
        dataset=dataset
    )
}

# Model initialization
if world.model_name not in MODELS:
    raise ValueError(f"Unknown model: {world.model_name}")
else:
    print(f"Selected model: {world.model_name}")

Recmodel = MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
print(f"Model initialized: {type(Recmodel)} on device {world.device}") # 디버깅 -> 삭제
