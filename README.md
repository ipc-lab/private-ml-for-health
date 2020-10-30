# Privacy Preserving Machine Learning for Healthcare Applications 

- A submission to [ITU AI/ML in 5G Challenge](https://www.itu.int/en/ITU-T/AI/challenge/2020/Pages/default.aspx) for [ITU-ML-5G-PS-022](https://sites.google.com/view/iitd5g/challenge-problems/privacy-preserving-aiml-in-5g-networks-for-healthcare-applications)

While a vast amount of valuable medical data are available in institutions distributed across states and countries, the users' privacy requirements are  strong barriers against utilizing the user's data for improving medical diagnostic models.  We propose a system that enables privacy-preserving training of deep neural networks on distributed medical images. We introduce a customization of differentially-private stochastic gradient descent for federated learning that, in combination with secure multi-party aggregation, achieves accurate model training while providing well-bounded privacy guarantees for users.  Experimental results on the diabetic retinopathy task shows that our nmethod provides a similar differential privacy guarantee as the centralized training counterpart, while achieving a better classification accuracy than federated learning with parallel differential privacy.

## Folders (Submission to ITU challenge):
 
1. `report`: includes the final report. For `1.Design document showing the reasons for the choice of privacy-preserving technique and the network architectural components.`
2. `private_training`: includes the source code and a JupyterNotebook tutorial for training the privacy-preserving model explained in the report. For `2.Source code for the implementation of the privacy-preserving design across various architectural components.`
3. `private_inference`: includes the source code and demo for running private inference on the trained model. For `3.Tested code and Test Report for all implementations- Implementations of Privacy-Preserving AI Technique, Trained Data Model, UI on smartphone.`
4. `video_demo`: include some video demos showing how to run training and inference. For  `4. A Video of the demonstration of Proof-of-Concept.`


# Live Demo:

Please use this link to get a private inference on a Diabetic Retinopathy medical image:

https://imperial-diagnostics.herokuapp.com/

