Training Model

To Train A Bayesian CNN run the TrainBayesianCNN.py

This has 4 preloaded datasets that will be trained with a bayesian LeNet

Model weights are stored in weights/
model history for train/val accuracy/loss is stored in model_history/

Will automatically train on every dataset in the dict

---
Plotting Training History

We can then run the plotTrainHistory.py to visualize the training history for the datasets

---

Generating Gradcam Masks

run generate_gradcam_outputs.py which will run monte carlo sampling on images from each class on the dataset.

will store input image, correct label, gradcam samples and the logits.

dataset parameters are hard coded into the python file

we are targetting the last layer

---

Computing Model Uncertainty on a Dataset

Run the CalculateUncertaintyForDataset.py code, this will take all the logits from the samples and compute the uncertainty

---

Understading gradcam outputs

run the parse_gradcam_outputs.py file to generate matplotlib charts for the gradcam samples

