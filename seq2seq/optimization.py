import tensorflow as tf


def update_varlist(loss, optimizer, var_list, grad_clip=5.0, global_step=None):
    gvs = optimizer.compute_gradients(loss, var_list=var_list)
    capped_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in gvs]
    update_step = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    return update_step


def build_optimization(model, optimization_params=None, loss=None):
    optimization_params = optimization_params or {}

    initial_lr = optimization_params.get("initial_lr", 1e-4)
    decay_steps = int(optimization_params.get("decay_steps", 100000))
    lr_decay = optimization_params.get("lr_decay", 0.999)
    grad_clip = optimization_params.get("grad_clip", 10.0)

    lr = tf.train.exponential_decay(
        initial_lr,
        model.global_step,
        decay_steps,
        lr_decay,
        staircase=True)

    model.loss = model.loss if model.loss is not None else loss
    model.optimizer = tf.train.AdamOptimizer(lr)

    model.train_op = update_varlist(
        model.loss, model.optimizer,
        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model.scope),
        grad_clip=grad_clip,
        global_step=model.global_step)
