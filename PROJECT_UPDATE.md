Project Update:

Step 1. Alpamayo-Inference - COMPLETE
We deployed Alpamayo-1.5 onto a remote GPU server and ran inference on 100 clips from the NVIDIA AI Physical Dataset.
For each clip we generated 16 candidate trajectories, and recorded which trajectory was closest to the ground truth.
Additionally, we recorded metrics like minADE and RMSE for comparison with the student model in Step 4.
The total inference time was around 12 hours on an NVIDIA RTX Quadro 8000, with the average GPU VRAM usage around 44.5GB

Step 2. Student Model
