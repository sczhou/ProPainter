# Weights

Put the downloaded pre-trained models to this folder.

The directory structure will be arranged as:
```
weights
   |- raft-things.pth
   |- recurrent_flow_completion.pth
   |- ProPainter.pth
   |- sam2/ (for the Gradio demo)
      |- sam2.1_hiera_small.pt (cached at startup by default)
      |- sam2.1_hiera_base_plus.pt (cached at startup by default)
      |- sam2.1_hiera_large.pt (cached at startup by default)
   |- i3d_rgb_imagenet.pt (for evaluating VFID metric)
   |- README.md
```
