from safetensors import safe_open
from collections import defaultdict

counts = defaultdict(int)
with safe_open('/Users/vanch/mlx-Ming-3.0-tts/models/Ming-3.0-tts-0.5b/model.safetensors', framework='pt', device='cpu') as f:
    keys = list(f.keys())
    print('Total tensor keys:', len(keys))
    for k in keys:
        parts = k.split('.')
        prefix = '.'.join(parts[:2]) if len(parts) > 1 else parts[0]
        counts[prefix] += 1
    
    print('Key prefix distribution:')
    for p, c in sorted(counts.items()):
        print(f'  {p:35s}: {c}')

    print('Non-model.model.layers keys:')
    for k in keys:
        if not k.startswith('model.model.layers.'):
            t = f.get_slice(k)
            print(f'  {k:50s} {t.get_shape()} {t.get_dtype()}')
