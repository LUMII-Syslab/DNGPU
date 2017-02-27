# DNGPU
Code for reproducing key results in the paper "Improving the Neural GPU Architecture for Algorithm Learning" by Karlis Freivalds, Renars Liepins.

The code demonstrates several improvements to the Neural GPU that substantially reduces training time and improves generalization. The improvements are: 1) hard nonlinearities with saturation cost; 2) diagonal gates that can be applied to active-memory models.

## Dependencies

This project currently requires the dev version of TensorFlow 0.10.0 available on Github: https://github.com/tensorflow/tensorflow/releases/tag/v0.10.0.

## Running Experiment

We provide the source code to run the training example:

```bash
python DNGPU_trainer.py
```
