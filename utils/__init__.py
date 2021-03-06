def is_using_gpu() -> bool:
    """Return if the script is being runned using a GPU.
    """
    # tf.test.is_gpu_available() is deprecated.
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')

    return len(gpus) > 0
