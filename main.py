import world
import utils
from world import cprint
from tensorboardX import SummaryWriter
import time
from os.path import join
import Procedure
import register

# ==============================
# SEED 설정
utils.set_seed(world.seed)
print(f">> SEED: {world.seed}")
# ==============================

# Dataset 및 모델 초기화
print(f">> Initializing dataset: {world.dataset}")
dataset = register.dataset  # register에서 dataset 참조

print(f">> Initializing model: {world.model_name}")
Recmodel = register.MODELS[world.model_name](
    world.config, dataset
)
Recmodel = Recmodel.to(world.device)
print(f">> Model initialized: {type(Recmodel)} on device {world.device}")

# 손실 함수 초기화
bpr = utils.BPRLoss(Recmodel, world.config)
print(f">> BPR Loss initialized with weight decay: {world.config['decay']}")

# 파일 경로 설정
weight_file = utils.getFileName()
embedding_file = f"{weight_file}_embeddings.pth"
print(f"Model weight file path: {weight_file}")
print(f"Model embedding file path: {embedding_file}")

# 모델 로드
if world.LOAD:
    try:
        Recmodel.load_model(weight_file, embedding_file)
        cprint(f"Loaded model weights and embeddings from {weight_file} and {embedding_file}")
    except FileNotFoundError:
        print(f"{weight_file} or {embedding_file} does not exist. Starting from scratch.")
Neg_k = 1

# TensorBoard 초기화
if world.tensorboard:
    tensorboard_path = join(
        str(world.BOARD_PATH),
        time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + str(world.comment)
    )
    writer = SummaryWriter(tensorboard_path)
    print(f"TensorBoard initialized at {tensorboard_path}")
else:
    writer = None
    cprint("TensorBoard is not enabled.")

# 훈련 루프
try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()

        # 매 10번째 epoch마다 테스트 수행
        if epoch % 10 == 0:
            cprint("[TEST]")
            Procedure.Test(
                dataset, Recmodel, epoch, writer, world.config.get('multicore', False)
            )

        # 한 epoch 동안 학습
        output_information = Procedure.BPR_train(
            dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=writer
        )
        print(f"EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}")

        # 모델 가중치 및 임베딩 저장
        Recmodel.save_model(weight_file, embedding_file)
        print(f"Saved model and embeddings at epoch {epoch + 1}")

finally:
    # TensorBoard 종료
    if writer:
        writer.close()
        print("TensorBoard writer closed.")

    # 훈련 종료 후 모델 최종 저장
    print("Saving model and embeddings after training...")
    Recmodel.save_model(weight_file, embedding_file)

print("Model training and saving completed.")
