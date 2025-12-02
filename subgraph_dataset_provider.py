import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow_gnn as tfgnn
from tensorflow_gnn import runner
import tensorflow as tf
from tensorflow_gnn.experimental import sampler
import functools
import itertools

from typing import Mapping

graph_schema = tfgnn.read_schema("graph_schema.pbtxt")

def create_sampling_model(full_graph_tensor: tfgnn.GraphTensor, sizes: Mapping[str, int]) -> tf.keras.Model:

    def edge_sampler(sampling_op: tfgnn.sampler.SamplingOp):    
        edge_set_name = sampling_op.edge_set_name
        sample_size = sizes[edge_set_name]
        return sampler.InMemUniformEdgesSampler.from_graph_tensor(
            full_graph_tensor,
            edge_set_name, sample_size=sample_size
        )
    def get_features(node_set_name: tfgnn.NodeSetName):
        return sampler.InMemIndexToFeaturesAccessor.from_graph_tensor(
            full_graph_tensor,
            node_set_name
        )

    sampling_spec_builder = tfgnn.sampler.SamplingSpecBuilder(graph_schema)
    seed = sampling_spec_builder.seed("product")
    products_bought_together = seed.sample(sizes["bought_together"], "bought_together", op_name="bt")
    sampling_spec = sampling_spec_builder.build()
    model = sampler.create_sampling_model_from_spec(graph_schema, sampling_spec, edge_sampler, get_features, seed_node_dtype=tf.int64)
    return model

class SubgraphDatasetProvider(runner.DatasetProvider):
    "Dataset provider"

    def __init__(self,
                full_graph_tensor: tfgnn.GraphTensor,
                sizes: Mapping[str, int],
                dataset: tf.data.Dataset,
                split_name: str,
                n: int):
        self._sampling_model = create_sampling_model(full_graph_tensor, sizes)
        self.input_graph_spec = self._sampling_model.output.spec
        self._seed_dataset = dataset
        self._split_name = split_name
        self._N = n
    def get_dataset(self, context: tf.distribute.InputContext) -> tf.data.Dataset:
        """Creates TF dataset"""
        ds = self._seed_dataset.shard(num_shards=context.num_input_pipelines, index = context.input_pipeline_id)
        # TODO Update to use correct number of training examples per type
        if self._split_name == "train":
            ds = ds.shuffle(self._N).repeat()
        ds = ds.batch(128)
        ds = ds.map(
            functools.partial(self.sample),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        return ds.unbatch().prefetch(tf.data.AUTOTUNE)

    def sample(self, seeds: tf.Tensor) -> tfgnn.GraphTensor:
        # seeds = tf.cast(seeds, tf.int32)
        batch_size = tf.size(seeds)
        # print(f"batch_size={batch_size}")
        seeds_ragged = tf.RaggedTensor.from_row_lengths(seeds, tf.ones([batch_size], dtype=tf.int64))
        print(seeds_ragged)
        return self._sampling_model(seeds_ragged)