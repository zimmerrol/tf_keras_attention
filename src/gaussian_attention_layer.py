from .gaussian_attention import gaussian_mask, gaussian_attention
from keras.layers import Layer

class VisualAttentionLayer(Layer):
    def __init__(self, output_dim, transpose=False, **kwargs):
        if len(output_dim) != 2:
            raise ValueError("`output_dim` has to be a 2D tensor [Height, Width].")
        self._output_dim = output_dim
        super(VisualAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(VisualAttentionLayer, self).build(input_shape)

    def call(self, x): 
        if len(x) != 2:
            raise ValueError("Input of the layer has to consist of 2 different inputs: the images and the parameters.")
        img_tensor, transform_params = x
        
        return gaussian_attention(img_tensor, transform_params, self._output_dim)

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2 and len(input_shape[0]) == 4:
            return (None, *self._output_dim, input_shape[0][-1])
        else:
            raise ValueError("The `input_shape` is not correct.")