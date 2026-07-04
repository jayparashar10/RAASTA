from logger import TrainingLogger

TrainingLogger.epoch_start(1, 20)

TrainingLogger.train(
    loss=0.84,
    dice=0.76,
    iou=0.62,
)

TrainingLogger.validation(
    loss=0.71,
    dice=0.82,
    iou=0.70,
)

TrainingLogger.checkpoint_saved()