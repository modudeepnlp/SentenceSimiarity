from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('classes', 2, 'classes')
flags.DEFINE_integer('dim', 100, 'dim')
flags.DEFINE_integer('out_dim', 100, 'out dim')
flags.DEFINE_integer('length', 70, 'max length')
flags.DEFINE_integer('epochs', 1, 'epochs')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('hidden_size', 32, 'hidden size')
flags.DEFINE_float('learning_rate', 1e-2, 'learning rate')
flags.DEFINE_float('dropout', 0.1, 'dropout')

