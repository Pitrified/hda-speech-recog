import math


def exp_decay_step(
    epoch: int,
    old_lr: float,
    initial_lrate: float = 0.001,
    drop: float = 0.4,
    epochs_drop: int = 15,
    min_lrate: float = 4e-5,
) -> float:
    """TODO: what is exp_decay doing?

    lrate = LearningRateScheduler(exp_decay_step)
    results = model.fit(... ,  callbacks=[lrate])
    """
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    if lrate < min_lrate:
        lrate = min_lrate

    print(f"Changing learning rate to {lrate}")
    return lrate


def exp_decay_smooth(
    epoch: int,
    old_lr: float,
    initial_lrate: float = 0.001,
    drop: float = 0.4,
    epochs_drop: int = 15,
    min_lrate: float = 4e-5,
) -> float:
    """TODO: what is exp_decay doing?

    lrate = LearningRateScheduler(exp_decay_smooth)
    results = model.fit(... ,  callbacks=[lrate])
    """
    lrate = initial_lrate * math.pow(drop, (1 + epoch) / epochs_drop)

    if lrate < min_lrate:
        lrate = min_lrate

    print(f"Changing learning rate to {lrate}")
    return lrate
