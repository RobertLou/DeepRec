# Embedding Variable Export Format

## Introduction

EmbeddingVariable is the expression of sparse parameters in DeepRec. After training the model, the user needs to obtain sparse parameter information from checkpoint, including: sparse features, feature frequency, and feature versions.

EmbeddingVariable is stored in ckpt in the form of tensor, assuming the name of the EV in the graph is a/b/c/embedding, then there are four tensors in checkpoint:

1. "a/b/c/embedding-keys" Tensor stores EV keys, shape=[N], dtype=int64;
2. "a/b/c/embedding-values" Tensor stores EV value，shape=[N, embedding_dim], dtype=float;
3. "a/b/c/embedding-freqs" Tenosr stores EV frequency，shape=[N]，dtype=int64;
4. "a/b/c/embedding-versions" Tensor stores the step number of the last update of EV, shape=[N], dtype=int64

You can read the value of checkpoint through the sdk of tensorflow to obtain the information corresponding to EV.

The first dimension shape of the above group of 4 tensors is the same, and they correspond in order. The EV of the partition is set, the number of tensors of the part depends on the set partitioner, and the sparse parameter of this feature is all tensor collections of all parts.

Execute the above code to get the following results:
![img_2.png](docs_zh/Embedding-Variable/img_2.jpg)


## Example

### Save Checkpoint

```python
import tensorflow as tf

a = tf.get_embedding_variable('a', embedding_dim=4)
b = tf.get_embedding_variable('b', embedding_dim=8, partitioner=tf.fixed_size_partitioner(4))

emb_a = tf.nn.embedding_lookup(a, tf.constant([0,1,2,3,4], dtype=tf.int64))
emb_b = tf.nn.embedding_lookup(b, tf.constant([5,6,7,8,9], dtype=tf.int64))

emb=tf.concat([emb_a, emb_b], axis=1)
loss=tf.reduce_sum(emb)

optimizer=tf.train.AdagradOptimizer(0.1)
train_op=optimizer.minimize(loss)

saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([emb, train_op]))
  saver.save(sess, "./ckpt_test/ckpt/model")
```

### Restore Checkpoint

```python
import tensorflow as tf
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.python.framework import meta_graph

ckpt_path='ckpt_test/ckpt'
path=ckpt_path+'/model.meta'
meta_graph = meta_graph.read_meta_graph_file(path)
ev_node_list=[]
for node in meta_graph.graph_def.node:
  if node.op == 'KvVarHandleOp':
    ev_node_list.append(node.name)

print("ev node list", ev_node_list)
# filter ev-slot
non_slot_ev_list=[]
for node in ev_node_list:
  if "Adagrad" not in node:
    non_slot_ev_list.append(node)
print("ev (exculde slot) node list", non_slot_ev_list)

for name in non_slot_ev_list:
  print(name+'-keys', tf.train.load_variable(ckpt_path, name+'-keys'))
  print(name+'-values', tf.train.load_variable(ckpt_path, name+'-values'))
  print(name+'-freqs', tf.train.load_variable(ckpt_path, name+'-freqs'))
  print(name+'-versions', tf.train.load_variable(ckpt_path, name+'-versions'))
```

### Result

```bash
root@i22b15440:/home/admin/chen.ding/gitlab/code/DeepRec# python ckpt_test/gen_ckpt.py
2022-04-18 08:56:37.462789: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2499445000 Hz
2022-04-18 08:56:37.468578: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7fd5bb36fc60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2022-04-18 08:56:37.469203: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[array([[ 0.19091757, -1.2494173 , -1.1098509 , -0.88375354,  1.5314401 ,
         0.05683817, -0.3698739 ,  1.7533029 , -0.95090204,  0.597882  ,
        -0.39382792,  1.2318059 ],
       [ 0.19091757, -1.2494173 , -1.1098509 , -0.88375354, -0.83169323,
         0.15894873, -0.66453475,  0.84301287,  1.125458  ,  0.12537971,
         0.7338474 , -0.02672509],
       [ 0.19091757, -1.2494173 , -1.1098509 , -0.88375354, -0.29247162,
        -1.1461716 ,  1.1172409 ,  1.9220417 , -1.2039331 , -1.248681  ,
        -1.9431682 , -0.27165115],
       [ 0.19091757, -1.2494173 , -1.1098509 , -0.88375354, -0.78701115,
        -0.28825614,  1.1483766 , -0.18648145,  0.7928211 , -0.4237969 ,
        -0.00831279,  0.68605185],
       [ 0.19091757, -1.2494173 , -1.1098509 , -0.88375354,  0.5981304 ,
         0.9115529 ,  1.2290154 ,  1.329322  ,  0.81167835,  0.43949136,
         0.4266789 ,  0.5895692 ]], dtype=float32), None]

root@i22b15440:/home/admin/chen.ding/gitlab/code/DeepRec# ls ckpt_test/ckpt/
checkpoint  model.data-00000-of-00001  model.index  model.meta

root@i22b15440:/home/admin/chen.ding/gitlab/code/DeepRec# python ckpt_test/read_ckpt.py
ev node list ['a', 'b/part_0', 'b/part_1', 'b/part_2', 'b/part_3', 'a/Adagrad', 'b/part_0/Adagrad', 'b/part_1/Adagrad', 'b/part_2/Adagrad', 'b/part_3/Adagrad']
ev (exculde slot) node list ['a', 'b/part_0', 'b/part_1', 'b/part_2', 'b/part_3']
a-keys [0 1 2 3 4]
a-values [[ 0.19091757 -1.2494173  -1.1098509  -0.88375354]
 [ 0.19091757 -1.2494173  -1.1098509  -0.88375354]
 [ 0.19091757 -1.2494173  -1.1098509  -0.88375354]
 [ 0.19091757 -1.2494173  -1.1098509  -0.88375354]
 [ 0.19091757 -1.2494173  -1.1098509  -0.88375354]]
a-freqs []
a-versions []
b/part_0-keys [8]
b/part_0-values [[-0.8823574  -0.3836024   1.0530304  -0.28182772  0.69747484 -0.51914316
  -0.10365905  0.5907056 ]]
b/part_0-freqs []
b/part_0-versions []
b/part_1-keys [5 9]
b/part_1-values [[ 1.4360939  -0.03850809 -0.46522018  1.6579567  -1.0462483   0.5025357
  -0.4891742   1.1364597 ]
 [ 0.50278413  0.81620663  1.1336691   1.2339758   0.7163321   0.3441451
   0.33133262  0.49422294]]
b/part_1-freqs []
b/part_1-versions []
b/part_2-keys [6]
b/part_2-values [[-0.83169323  0.15894873 -0.66453475  0.84301287  1.125458    0.12537971
   0.7338474  -0.02672509]]
b/part_2-freqs []
b/part_2-versions []
b/part_3-keys [7]
b/part_3-values [[-0.3878179  -1.2415178   1.0218947   1.8266954  -1.2992793  -1.3440272
  -2.0385144  -0.36699742]]
b/part_3-freqs []
b/part_3-versions []
```
