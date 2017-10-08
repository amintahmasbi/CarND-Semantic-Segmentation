## guide on how to optimize for inference


The `optimize_for_inference` module takes a `frozen binary GraphDef` file as input and outputs the `optimized Graph Def` file which you can use for inference. And to get the `frozen binary GraphDef file` you need to use the module `freeze_graph` which takes a` GraphDef proto`, a `SaverDef proto` and a set of variables stored in a checkpoint file. The steps to achieve that is given below:

###1. Saving a tensorflow graph
```
 # make and save a simple graph
 G = tf.Graph()
 with G.as_default():
   x = tf.placeholder(dtype=tf.float32, shape=(), name="x")
   a = tf.Variable(5.0, name="a")
   y = tf.add(a, x, name="y")
   saver = tf.train.Saver()

with tf.Session(graph=G) as sess:
   sess.run(tf.global_variables_initializer())
   out = sess.run(fetches=[y], feed_dict={x: 1.0})

  # Save GraphDef
  tf.train.write_graph(sess.graph_def,'.','my_graph.pb')
  # Save checkpoint
  saver.save(sess=sess, save_path="test_model")
``` 
__Note:__ Check the `main.py` of _CarND-Semantic-Segmentation_ project for the detailed implementation of tensors.

---
###2. Freeze graph

```
python -m tensorflow.python.tools.freeze_graph \
--input_graph=my_graph.pb \
--input_checkpoint=model.ckpt-49 \
--input_binary=true \
--output_graph=my_frozen_graph.pb \
--output_node_names=Reshape_3 
```
__Note:__ use `cross_entropy_loss` instead of Reshape_3 (extracted from tensorboard)

---
###3. Optimize for inference
```
python -m tensorflow.python.tools.optimize_for_inference \
--input=my_frozen_graph.pb \
--output=my_optimized_graph.pb \
--frozen_graph=True \
--input_names=image_input \
--output_names=Reshape_3

```

__Note:__ use `cross_entropy_loss` instead of Reshape_3 (extracted from tensorboard)

---

###4. Check number of Ops in graph
```
python check_optimization.py
```

---

###5. Using the Optimized graph

####1. JIT & AOT

Just In Time (JIT) and Ahead Of Time (AOT) compilation

```
# Create a TensorFlow configuration object. This will be 
# passed as an argument to the session.
config = tf.Config()
# JIT level, this can be set to ON_1 or ON_2 
jit_level = tf.OptimizerOptions.ON_1
config.graph_options.optimizer_options.global_jit_level = jit_level
```
That’s it! All that’s left to be done is pass the config into the session:
```
with tf.Session(config=config) as sess:
```
####2. Reusing the Graph
```
with tf.gfile.GFile('my_optimized_graph.pb', 'rb') as f:
   graph_def_optimized = tf.GraphDef()
   graph_def_optimized.ParseFromString(f.read())

G = tf.Graph()

with tf.Session(graph=G) as sess:
    y, = tf.import_graph_def(graph_def_optimized, return_elements=['y:0'])
    print('Operations in Optimized Graph:')
    print([op.name for op in G.get_operations()])
    x = G.get_tensor_by_name('import/x:0')
    tf.global_variables_initializer().run()
    out = sess.run(y, feed_dict={x: 1.0})
    print(out)

#Output
#Operations in Optimized Graph:
#['import/x', 'import/a', 'import/y']
#6.0
```

__Note:__ variable names are only for reference  (different from the implementation)

__Another method:__
```
from graph_utils import load_graph

sess, _ = load_graph(‘your_graph.pb’)
graph = sess.graph

image_input = graph.get_tensor_by_name('image_input:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
#softmax = graph.get_tensor_by_name('Softmax:0')
softmax = graph.get_tensor_by_name('Reshape_3:0')

probs = sess.run(softmax, {image_input: img, keep_prob: 1.0})
```
__Note:__ use `cross_entropy_loss` instead of Reshape_3 (extracted from tensorboard)

###6. (OPTIONAL) 8-bit Quantization

- Download tensorflow source from `github`

- Install `Bazel` if it is not already installed

- `cd` to tensorflow folder 

- might need to run `./configure`
```
bazel build tensorflow/tools/graph_transforms:transform_graph
```
- `cd` back to model directory
```
~/git/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=my_frozen_graph.pb \
--out_graph=my_eightbit_graph.pb \
--inputs=image_input \
--outputs=Reshape_3 \
--transforms='
add_default_attributes
remove_nodes(op=Identity, op=CheckNumerics)
fold_constants(ignore_errors=true)
fold_batch_norms
fold_old_batch_norms
fuse_resize_and_conv
quantize_weights
quantize_nodes
strip_unused_nodes
sort_by_execution_order'

```

__Note:__ use `cross_entropy_loss` instead of Reshape_3 (extracted from tensorboard)












