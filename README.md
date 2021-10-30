# Shapenet-pyrender
This is a compact implementation of a batched OBJ-renderer in pyrender. The inspiration was drawn
from the "Stanford Shapenet Renderer". This code can be used to render RGB-D images with ground truth masks of shapenet models.

To render a batch of obj files in parallel, use the "find" command in conjunction with xargs:

find /path/to/shapenet/folder -name *.obj -print0 | xargs -0 -n1 -P1 -I {} python pyrenderer.py --output_dir ./output/ --mesh_fpath {} --num_observations 25 --sphere_radius 1 --mode=train 
