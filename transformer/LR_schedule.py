import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, stopped_at=0, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.stopped_at = int(stopped_at)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(int(step)+self.stopped_at, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'd_model': tf.get_static_value(self.d_model),
            'warmup_steps': self.warmup_steps,
            'stopped_at': tf.get_static_value(self.stopped_at),
        }
        return config
