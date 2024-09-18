# GOTHAM
Generative cOnditional Transformer for Halo's Auto-regressive Modeling

## Description
The code aims to generate mock N-body-like halo catalog from a density field from low-resolution FAST-PM simulation by treating it like a language translation problem and using large-languge model-like architecture. It extracts features from the PM simulation to use as a conditioning to the cross-attention blocks of transformers and learns the tokens corresponding to halo properties. Currently it predicts 3D halo positions and masses but can be extended to predict their velocities and concentrations.



