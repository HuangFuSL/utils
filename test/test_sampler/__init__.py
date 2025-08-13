import os

for _ in os.listdir(os.path.join(os.path.dirname(__file__), '../../sampler')):
    for ext in ['.so', '.pyd', '.dll', '.dylib']:
        if _.endswith(ext):
            os.remove(os.path.join(os.path.dirname(__file__), '../../sampler', _))

import sampler
