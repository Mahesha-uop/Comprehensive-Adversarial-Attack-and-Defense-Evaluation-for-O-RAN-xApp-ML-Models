import numpy as np
from PIL import Image
import os, glob

for folder in ['soi', 'cwi']:
    imgs = glob.glob(f"newdataset/{folder}/*.png")[:5]
    for p in imgs[:1]:
        img = Image.open(p)
        arr = np.array(img)
        print(f"{folder}: shape={arr.shape}, mode={img.mode}, "
              f"min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")
        # Check if RGB channels differ
        if len(arr.shape) == 3 and arr.shape[2] >= 3:
            r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
            print(f"  R: [{r.min()}-{r.max()}], G: [{g.min()}-{g.max()}], B: [{b.min()}-{b.max()}]")
            print(f"  Channels identical? R==G: {np.array_equal(r,g)}, R==B: {np.array_equal(r,b)}")