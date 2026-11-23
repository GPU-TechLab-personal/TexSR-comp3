import os, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
import tensorflow as tf
print("TF", tf.__version__, "built CUDA", tf.sysconfig.get_build_info().get("cuda_version"), "cudnn", tf.sysconfig.get_build_info().get("cudnn_version"))
g = tf.config.list_physical_devices("GPU"); print("GPUs", g)
with tf.device("/GPU:0"):
    t = time.time()
    x = tf.random.normal([8, 128, 128, 3])
    conv = tf.keras.layers.Conv2D(64, 3, padding="same")
    with tf.GradientTape() as tape:
        y = conv(x); loss = tf.reduce_mean(tf.square(y))
    gr = tape.gradient(loss, conv.trainable_variables)
    m = tf.matmul(tf.random.normal([1024, 1024]), tf.random.normal([1024, 1024]))
    print("conv+grad+matmul OK", float(loss), float(tf.reduce_sum(m)), "first-call %.1fs" % (time.time() - t))
    t = time.time(); y = conv(x); m = tf.matmul(m, m); _ = float(tf.reduce_sum(m)); print("second call %.3fs" % (time.time() - t))
