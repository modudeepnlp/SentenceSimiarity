# 2019-05-24 - baseline:
  - config: { "n_embed": len(vocab), "d_embed": 32, "n_output": 3, "n_epoch": 10, n_batch": 256, "n_layer": 1, "cells": 2, "dropout": 0.1 ## HBMP }
  - dev:  66.7
  - test: 66.4


# 2019-06-02 - baseline + (p;h;|p-h|;p*h):
  - config: { "n_embed": len(vocab), "d_embed": 32, "n_output": 3, "n_epoch": 10, n_batch": 256, "n_layer": 1, "cells": 2, "dropout": 0.1 ## HBMP }
  - dev:  75.3
  - test: 75.8


# 2019-06-07 - Prevous + HBMP:
  - config: { "n_embed": len(vocab), "d_embed": 32, "n_output": 3, "n_epoch": 10, n_batch": 256, "n_layer": 1, "cells": 2, "dropout": 0.1 ## HBMP }
  - dev:  77.6
  - test: 77.5


# 2019-06-07 - Prevous + layer3:
  - config: { "n_embed": len(vocab), "d_embed": 32, "n_output": 3, "n_epoch": 10, n_batch": 256, "n_layer": 1, "cells": 2, "dropout": 0.1 ## HBMP }
  - dev:  77.3
  - test: 77.2


# 2019-06-07 - Prevous + RELU(2):
  - config: { "n_embed": len(vocab), "d_embed": 32, "n_output": 3, "n_epoch": 10, n_batch": 256, "n_layer": 1, "cells": 2, "dropout": 0.1 ## HBMP }
  - dev:  78.6
  - test: 78.6
