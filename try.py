# import tensorflow as tf

# # TFRecord 文件路径
# tfrecord_path = "/workspace/bridge_orig/1.0.0/bridge_dataset-train.tfrecord-00000-of-01024"

# # 创建 TFRecordDataset
# dataset = tf.data.TFRecordDataset([tfrecord_path])

# # 解析单个 tf.train.Example
# def parse_example(serialized_example):
#     # 解析 Example，获取特征字典
#     example = tf.train.Example()
#     example.ParseFromString(serialized_example.numpy())
    
#     # 打印字段名称
#     feature_dict = example.features.feature
#     print("Fields in tf.train.Example:", list(feature_dict.keys()))
    
#     # 打印第一个样本的完整内容
#     print("Sample content:", example)
    
#     return example

# # 读取第一个样本并解析
# for raw_record in dataset.take(1):
#     parse_example(raw_record)
#     break

import ray
ray.init()
actors = ray.state.actors()  # 旧版 Ray 的方法
for actor_id, actor_info in actors.items():
    print(f"Actor ID: {actor_id}, Name: {actor_info.get('name')}, Namespace: {actor_info.get('namespace')}")