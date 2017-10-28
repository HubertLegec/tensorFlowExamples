import tensorflow as tf

# download and read in data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# --- MODEL ---

# We want to be able to input any number of MNIST images, each flattened into a 784-dimensional vector.
# We represent this as a 2-D tensor of floating-point numbers, with a shape [None, 784]
x = tf.placeholder(tf.float32, [None, 784])

# Weights and biases for model
# Notice that W has a shape of [784, 10] because we want to multiply the 784-dimensional image vectors
# by it to produce 10-dimensional vectors of evidence for the difference classes
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b

# --- TRAINING ---

# correct answers
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
