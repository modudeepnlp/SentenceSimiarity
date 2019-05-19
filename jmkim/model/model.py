from tensorflow.python.keras import layers
from tensorflow.python.keras import backend


class ManDist(layers):
	def __init__(self, **kwargs):
		self.result = None
		super(ManDist, self).__init__(**kwargs)

	def build(self, input_shape):
		super(ManDist, self).build(input_shape)

	def call(self, x, **kwargs):
		self.result = backend.exp(-backend.sum(backend.abs(x[0] - x[1]), axis=1, keepdims=True))
		return self.result

	def compute_output_shape(self, input_shape):
		return backend.int_shape(self.result)
