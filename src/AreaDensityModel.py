import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras.layers import Input, Layer, Dropout, LSTM,GRU, Dense, BatchNormalization,Permute, Reshape,Concatenate
from stellargraph.mapper import SlidingFeaturesNodeGenerator
from stellargraph.core.experimental import experimental
from stellargraph.core.utils import calculate_laplacian


class FixedAdjacencyGraphConvolution(Layer):

    """
    Graph Convolution (GCN) Keras layer.
    The implementation is based on https://github.com/tkipf/keras-gcn.
    Original paper: Semi-Supervised Classification with Graph Convolutional Networks. Thomas N. Kipf, Max Welling,
    International Conference on Learning Representations (ICLR), 2017 https://github.com/tkipf/gcn
    Notes:
      - The inputs are 3 dimensional tensors: batch size, sequence length, and number of nodes.
      - This class assumes that a simple unweighted or weighted adjacency matrix is passed to it,
        the normalized Laplacian matrix is calculated within the class.
    Args:
        units (int): dimensionality of output feature vectors
        A (N x N): weighted/unweighted adjacency matrix
        activation (str or func): nonlinear activation applied to layer's output to obtain output features
        use_bias (bool): toggles an optional bias
        kernel_initializer (str or func, optional): The initialiser to use for the weights.
        kernel_regularizer (str or func, optional): The regulariser to use for the weights.
        kernel_constraint (str or func, optional): The constraint to use for the weights.
        bias_initializer (str or func, optional): The initialiser to use for the bias.
        bias_regularizer (str or func, optional): The regulariser to use for the bias.
        bias_constraint (str or func, optional): The constraint to use for the bias.
    """

    def __init__(
        self,
        units,
        A,
        activation=None,
        use_bias=True,
        input_dim=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs,
    ):
        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = (input_dim,)

        self.units = units
        self.adj = calculate_laplacian(A) if A is not None else None

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        super().__init__(**kwargs)

    def get_config(self):
        """
        Gets class configuration for Keras serialization.
        Used by Keras model serialization.
        Returns:
            A dictionary that contains the config of the layer
        """

        config = {
            "units": self.units,
            "use_bias": self.use_bias,
            "activation": activations.serialize(self.activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            # the adjacency matrix argument is required, but
            # (semi-secretly) supports None for loading from a saved
            # model, where the adjacency matrix is a saved weight
            "A": None,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape of the layer.
        Assumes the following inputs:
        Args:
            input_shapes (tuple of int)
                Shape tuples can include None for free dimensions, instead of an integer.
        Returns:
            An input shape tuple.
        """
        feature_shape = input_shapes

        return feature_shape[0], feature_shape[1], self.units

    def build(self, input_shapes):
        """
        Builds the layer
        Args:
            input_shapes (list of int): shapes of the layer's inputs (the batches of node features)
        """
        _batch_dim, n_nodes, features = input_shapes

        if self.adj is not None:
            adj_init = initializers.constant(self.adj)
        else:
            adj_init = initializers.zeros()

        self.A = self.add_weight(
            name="A", shape=(n_nodes, n_nodes), trainable=False, initializer=adj_init
        )
        self.kernel = self.add_weight(
            shape=(features, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                # ensure the per-node bias can be broadcast across each feature
                shape=(n_nodes, 1),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, features):
        """
        Applies the layer.
        Args:
            features (ndarray): node features (size B x N x F), where B is the batch size, F = TV is
                the feature size (consisting of the sequence length and the number of variates), and
                N is the number of nodes in the graph.
        Returns:
            Keras Tensor that represents the output of the layer.
        """

        # Calculate the layer operation of GCN
        # shape = B x F x N
        nodes_last = tf.transpose(features, [0, 2, 1])
        neighbours = K.dot(nodes_last, self.A)

        # shape = B x N x F
        h_graph = tf.transpose(neighbours, [0, 2, 1])
        # shape = B x N x units
        output = K.dot(h_graph, self.kernel)

        # Add optional bias & apply activation
        if self.bias is not None:
            output += self.bias

        output = self.activation(output)

        return output

class GraphConvolution(Layer):

    """
    Graph Convolution (GCN) Keras layer.
    The implementation is based on https://github.com/tkipf/keras-gcn.
    Original paper: Semi-Supervised Classification with Graph Convolutional Networks. Thomas N. Kipf, Max Welling,
    International Conference on Learning Representations (ICLR), 2017 https://github.com/tkipf/gcn
    Notes:
      - The batch axis represents independent graphs to be convolved with this GCN kernel (for
        instance, for full-batch node prediction on a single graph, its dimension should be 1).
      - If the adjacency matrix is dense, both it and the features should have a batch axis, with
        equal batch dimension.
      - If the adjacency matrix is sparse, it should not have a batch axis, and the batch
        dimension of the features must be 1.
      - There are two inputs required, the node features,
        and the normalized graph Laplacian matrix
      - This class assumes that the normalized Laplacian matrix is passed as
        input to the Keras methods.
    .. seealso:: :class:`.GCN` combines several of these layers.
    Args:
        units (int): dimensionality of output feature vectors
        activation (str or func): nonlinear activation applied to layer's output to obtain output features
        use_bias (bool): toggles an optional bias
        final_layer (bool): Deprecated, use ``tf.gather`` or :class:`.GatherIndices`
        kernel_initializer (str or func, optional): The initialiser to use for the weights.
        kernel_regularizer (str or func, optional): The regulariser to use for the weights.
        kernel_constraint (str or func, optional): The constraint to use for the weights.
        bias_initializer (str or func, optional): The initialiser to use for the bias.
        bias_regularizer (str or func, optional): The regulariser to use for the bias.
        bias_constraint (str or func, optional): The constraint to use for the bias.
    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        final_layer=None,
        input_dim=None,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        **kwargs,
    ):
        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = (input_dim,)

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        if final_layer is not None:
            raise ValueError(
                "'final_layer' is not longer supported, use 'tf.gather' or 'GatherIndices' separately"
            )

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        super().__init__(**kwargs)

    def get_config(self):
        """
        Gets class configuration for Keras serialization.
        Used by Keras model serialization.
        Returns:
            A dictionary that contains the config of the layer
        """

        config = {
            "units": self.units,
            "use_bias": self.use_bias,
            "activation": activations.serialize(self.activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape of the layer.
        Assumes the following inputs:
        Args:
            input_shapes (tuple of int)
                Shape tuples can include None for free dimensions, instead of an integer.
        Returns:
            An input shape tuple.
        """
        feature_shape, *As_shapes = input_shapes

        batch_dim = feature_shape[0]
        out_dim = feature_shape[1]

        return batch_dim, out_dim, self.units

    def build(self, input_shapes):
        """
        Builds the layer
        Args:
            input_shapes (list of int): shapes of the layer's inputs (node features and adjacency matrix)
        """
        feat_shape = input_shapes[0]
        input_dim = int(feat_shape[-1])

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        """
        Applies the layer.
        Args:
            inputs (list): a list of 2 input tensors that includes
                node features (size 1 x N x F),
                graph adjacency matrix (size N x N),
                where N is the number of nodes in the graph, and
                F is the dimensionality of node features.
        Returns:
            Keras Tensor that represents the output of the layer.
        """
        features, *As = inputs

        # Calculate the layer operation of GCN
        A = As[0]
        if K.is_sparse(A):
            # FIXME(#1222): batch_dot doesn't support sparse tensors, so we special case them to
            # only work with a single batch element (and the adjacency matrix without a batch
            # dimension)
            if features.shape[0] != 1:
                raise ValueError(
                    f"features: expected batch dimension = 1 when using sparse adjacency matrix in GraphConvolution, found features batch dimension {features.shape[0]}"
                )
            if len(A.shape) != 2:
                raise ValueError(
                    f"adjacency: expected a single adjacency matrix when using sparse adjacency matrix in GraphConvolution (tensor of rank 2), found adjacency tensor of rank {len(A.shape)}"
                )

            features_sq = K.squeeze(features, axis=0)
            h_graph = K.dot(A, features_sq)
            h_graph = K.expand_dims(h_graph, axis=0)
        else:
            h_graph = K.batch_dot(A, features)
        output = K.dot(h_graph, self.kernel)

        # Add optional bias & apply activation
        if self.bias is not None:
            output += self.bias
        output = self.activation(output)

        return output
# @experimental(
#     reason="Lack of unit tests and code refinement", issues=[1132, 1526, 1564]
# )
class GCN_GRU:

    """
    GCN_GRU is a univariate timeseries forecasting method. The architecture  comprises of a stack of N1 Graph Convolutional layers followed by N2 GRU layers, a Dropout layer, and  a Dense layer.
    This main components of GNN architecture is inspired by: T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction (https://arxiv.org/abs/1811.05320).
    The implementation of the above paper is based on one graph convolution layer stacked with a GRU layer.
    The StellarGraph implementation is built as a stack of the following set of layers:
    1. User specified no. of Graph Convolutional layers
    2. User specified no. of GRU layers
    3. 1 Dense layer
    4. 1 Dropout layer.
    The last two layers consistently showed better performance and regularization experimentally.
    .. seealso::
       Example using GCN_GRU: `spatio-temporal time-series prediction <https://stellargraph.readthedocs.io/en/stable/demos/time-series/gcn-GRU-time-series.html>`__.
       Appropriate data generator: :class:`.SlidingFeaturesNodeGenerator`.
       Related model: :class:`.GCN` for graphs without time-series node features.
    Args:
       seq_len: No. of GRU cells
       adj: unweighted/weighted adjacency matrix of [no.of nodes by no. of nodes dimension
       gc_layer_sizes (list of int): Output sizes of Graph Convolution  layers in the stack.
       GRU_layer_sizes (list of int): Output sizes of GRU layers in the stack.
       generator (SlidingFeaturesNodeGenerator): A generator instance.
       bias (bool): If True, a bias vector is learnt for each layer in the GCN model.
       dropout (float): Dropout rate applied to input features of each GCN layer.
       gc_activations (list of str or func): Activations applied to each layer's output; defaults to ``['relu', ..., 'relu']``.
       GRU_activations (list of str or func): Activations applied to each layer's output; defaults to ``['tanh', ..., 'tanh']``.
       kernel_initializer (str or func, optional): The initialiser to use for the weights of each layer.
       kernel_regularizer (str or func, optional): The regulariser to use for the weights of each layer.
       kernel_constraint (str or func, optional): The constraint to use for the weights of each layer.
       bias_initializer (str or func, optional): The initialiser to use for the bias of each layer.
       bias_regularizer (str or func, optional): The regulariser to use for the bias of each layer.
       bias_constraint (str or func, optional): The constraint to use for the bias of each layer.
     """

    def __init__(
        self,
        seq_len,
        adj,
        gc_layer_sizes,
        GRU_layer_sizes,
        batch_size=None,
        gc_activations=None,
        generator=None,
        GRU_activations=None,
        bias=True,
        dropout=0.5,
        kernel_initializer=None,
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer=None,
        bias_regularizer=None,
        bias_constraint=None,
    ):
        if generator is not None:
            if not isinstance(generator, SlidingFeaturesNodeGenerator):
                raise ValueError(
                    f"generator: expected a SlidingFeaturesNodeGenerator, found {type(generator).__name__}"
                )

            if seq_len is not None or adj is not None:
                raise ValueError(
                    "expected only one of generator and (seq_len, adj) to be specified, found multiple"
                )

            adj = generator.graph.to_adjacency_matrix(weighted=True).todense()
            seq_len = generator.window_size
            variates = generator.variates
        else:
            variates = None

        super(GCN_GRU, self).__init__()

        n_gc_layers = len(gc_layer_sizes)
        n_GRU_layers = len(GRU_layer_sizes)

        self.GRU_layer_sizes = GRU_layer_sizes
        self.gc_layer_sizes = gc_layer_sizes
        self.bias = bias
        self.dropout = dropout
        self.adj = adj
        self.n_nodes = 16
        self.n_features = seq_len
        self.seq_len = seq_len
        self.multivariate_input = True#truevariates is not None
        self.variates = 3 #variates if self.multivariate_input else 1
        self.outputs = self.n_nodes * self.variates
        self.batch_size=batch_size
        self.covs=24
        self.n_sus_layers=6
        self.n_areaden_layers=12
        self.n_count_layers=8
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        # Activation function for each gcn layer
        if gc_activations is None:
            gc_activations = ["relu"] * n_gc_layers
        elif len(gc_activations) != n_gc_layers:
            raise ValueError(
                "Invalid number of activations; require one function per graph convolution layer"
            )
        self.gc_activations = gc_activations

        # Activation function for each GRU layer
        if GRU_activations is None:
            GRU_activations = ["tanh"] * n_GRU_layers
        elif len(GRU_activations) != n_GRU_layers:
            padding_size = n_GRU_layers - len(GRU_activations)
            if padding_size > 0:
                GRU_activations = GRU_activations + ["tanh"] * padding_size
            else:
                raise ValueError(
                    "Invalid number of activations; require one function per GRU layer"
                )
        self.GRU_activations = GRU_activations
        self._GRU_layers = [
            GRU(layer_size, activation=activation, return_sequences=True)
            for layer_size, activation in zip(
                self.GRU_layer_sizes[:-1], self.GRU_activations
            )
        ]
        self._GRU_layers.append(
            GRU(
                self.GRU_layer_sizes[-1],
                activation=self.GRU_activations[-1],
                return_sequences=False,
            )
        )
        self._gc_layers = [
            FixedAdjacencyGraphConvolution(
                units=self.variates * layer_size,
                A=self.adj,
                activation=activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint,
            )
            for layer_size, activation in zip(self.gc_layer_sizes, self.gc_activations)
        ]
        # self._sus_layers=[Dense(32, activation="selu"), BatchNormalization(),Dropout(rate=0.2),
        # Dense(32, activation="selu"), BatchNormalization(),Dropout(rate=0.2),Dense(32, activation="selu"), BatchNormalization(),Dropout(rate=0.2),
        # Dense(32, activation="selu"), BatchNormalization(),Dropout(rate=0.2),Dense(32, activation="selu"), BatchNormalization(),Dropout(rate=0.2),
        # ]
        # self._areaden_layers=[Dense(32, activation="selu"), BatchNormalization(),Dropout(rate=0.2),
        # Dense(32, activation="selu"), BatchNormalization(),Dropout(rate=0.2),Dense(32, activation="selu"), BatchNormalization(),Dropout(rate=0.2),
        # Dense(32, activation="selu"), BatchNormalization(),Dropout(rate=0.2),Dense(32, activation="selu"), BatchNormalization(),Dropout(rate=0.2),
        # ]
        # self._count_layers=[Dense(32, activation="selu"), BatchNormalization(),Dropout(rate=0.2),
        # Dense(32, activation="selu"), BatchNormalization(),Dropout(rate=0.2),Dense(32, activation="selu"), BatchNormalization(),Dropout(rate=0.2),
        # Dense(32, activation="selu"), BatchNormalization(),Dropout(rate=0.2),Dense(32, activation="selu"), BatchNormalization(),Dropout(rate=0.2),
        # ]

        self._decoder_layer_sus = Dense(1, activation="sigmoid",name='sus')
        self._decoder_layer_area = Dense(1, activation="relu",name='area')
        self._decoder_layer_count = Dense(1, activation="exponential",name='count')

    def __call__(self, x,adj_t,cov):
        x_in = x

        h_layer = x_in
        h_layer=tf.reshape(h_layer,(-1,self.seq_len,self.variates))#Reshape((-1,self.seq_len,self.variates))(h_layer)
      
        for layer_size, activation in zip(self.GRU_layer_sizes[:-1], self.GRU_activations[:-1]):
            h_layer=GRU(layer_size,activation=activation,return_sequences=True)(h_layer)
        h_layer=GRU(self.GRU_layer_sizes[-1],activation=self.GRU_activations[-1],return_sequences=False)(h_layer)
        h_layer=tf.reshape(h_layer,(-1,self.n_nodes,128))
        h_layer=Concatenate()([h_layer,cov])

        for layer_size, activation in zip(self.gc_layer_sizes, self.gc_activations):
            h_layer=GraphConvolution(
                units=self.variates * layer_size,
        
                activation=activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint,
                )([h_layer,adj_t])
        sus_layer=h_layer
        area_layer=h_layer
        count_layer=h_layer
        # for i in range(self.n_sus_layers):
        #     sus_layer=Dense(32, activation="relu",name=f'sus_DEN_{str(i)}')(sus_layer)
        #     sus_layer=BatchNormalization(name=f'sus_BN_{str(i)}')(sus_layer)
        #     sus_layer=Dropout(rate=0.2,name=f'sus_DR_{str(i)}')(sus_layer)
        for i in range(self.n_areaden_layers):
            area_layer=Dense(64, activation="relu",name=f'area_DEN_{str(i)}')(area_layer)
            area_layer=BatchNormalization(name=f'area_BN_{str(i)}')(area_layer)
            area_layer=Dropout(rate=0.2,name=f'area_DR_{str(i)}')(area_layer)
        # for i in range(self.n_count_layers):
        #     count_layer=Dense(32, activation="relu",name=f'count_DEN_{str(i)}')(count_layer)
        #     count_layer=BatchNormalization(name=f'count_BN_{str(i)}')(count_layer)
        #     count_layer=Dropout(rate=0.2,name=f'count_DR_{str(i)}')(count_layer)

        # sus = self._decoder_layer_sus(sus_layer)
        area = self._decoder_layer_area(area_layer)
        # count = self._decoder_layer_count(count_layer)
        return area#[sus,area,count]

     

    def in_out_tensors(self):

        # Inputs for features
        if self.multivariate_input:
            shape = (self.batch_size, self.n_nodes, self.n_features, self.variates)
            shape_cov=(self.batch_size, self.n_nodes, self.covs)
        else:
            shape = (self.batch_size, self.n_nodes, self.n_features)

        x_t = Input(batch_shape=shape)
        c_inp=Input(batch_shape=shape_cov)

        # Indices to gather for model output
        adj_t = Input(batch_shape=(None, self.n_nodes,self.n_nodes), dtype="float32")

        x_inp = x_t
        x_out = self(x_inp,adj_t,c_inp)

        return [x_inp,adj_t,c_inp], x_out