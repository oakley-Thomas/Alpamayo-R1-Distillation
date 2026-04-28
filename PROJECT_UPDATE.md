Project Update:

Step 1. Alpamayo-Inference - COMPLETE
We deployed Alpamayo-1.5 onto a Quadro RTX 8000 machine and ran inference on 100 clips from the NVIDIA Physical AI dataset.
For each clip we generated 16 candidate trajectories, and recorded which trajectory was closest to the ground truth.
Additionally, we recorded metrics like minADE and RMSE for comparison with the student model in Step 4.
The total inference time was around 12 hours on an NVIDIA RTX Quadro 8000, with the average GPU VRAM usage around 44.5GB

Step 2. Student Model
By April 26th,we want to select a lightweight backbone like Phi-3 Mini and implement an MLP action head to begin the student model training. Spend the following week (April 27 – May 3) running the distillation loop using a loss function that balances your teacher's 16 candidate trajectories with ground truth labels. From May 4th to May 8th, benchmark the student’s minADE, RMSE, and inference speed against your teacher baseline to generate comparison tables. Use the final days before May 11th to visualize trajectory overlays and document how much performance was traded for the significantly lower VRAM footprint.
