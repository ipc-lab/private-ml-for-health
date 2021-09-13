# Dopamine: Differentially Private Federated Learning on Medical Data

- Please read the paper here: https://arxiv.org/abs/2101.11693 

- Chosen as the best submission to [ITU AI/ML in 5G Challenge](https://www.itu.int/en/ITU-T/AI/challenge/2020/Pages/default.aspx) for [ITU-ML-5G-PS-022](https://sites.google.com/view/iitd5g/challenge-problems/privacy-preserving-aiml-in-5g-networks-for-healthcare-applications)


## Abstract 
While rich medical datasets are hosted in hospitals distributed across the world, concerns on patients' privacy is a barrier against using such data to train deep neural networks (DNNs) for medical diagnostics. We propose Dopamine, a system to train DNNs on distributed datasets, which employs federated learning (FL) with differentially-private stochastic gradient descent (DPSGD), and, in combination with secure aggregation, can establish a better trade-off between differential privacy (DP) guarantee and DNN's accuracy than other approaches. Results on a diabetic retinopathy~(DR) task show that Dopamine provides a DP guarantee close to the centralized training counterpart, while achieving a better classification accuracy than FL with parallel DP where DPSGD is applied without coordination. 

## Folders:
 
1. `report`: includes the final report. For `1.Design document showing the reasons for the choice of privacy-preserving technique and the network architectural components.`
2. `private_training`: includes the source code and a JupyterNotebook tutorial for training the privacy-preserving model explained in the report. For `2.Source code for the implementation of the privacy-preserving design across various architectural components.`
3. `private_inference`: includes the source code and demo for running inference on the privately trained model. For `3.Tested code and Test Report for all implementations- Implementations of Privacy-Preserving AI Technique, Trained Data Model, UI on smartphone.`
4. `video_demo`: include some video demos showing how to run training and inference. For  `4. A Video of the demonstration of Proof-of-Concept.`


## Tutorial

We provided a Jupyter Notebook for training on Google Colab. Please see the file `JNotebook_running_FSCDP_on_Colab.ipynb` in the `private_training` folder.

## Live Demo:

Please use this link to get an inference on a Diabetic Retinopathy medical image:

https://imperial-diagnostics.herokuapp.com/

(Note: implementing the pure private inference is still in progress...)

## Citation
If you find the provided code or the proposed algorithms useful, please cite this work as:
```bibtex
@article{malekzadeh2021dopamine,
  title={Dopamine: Differentially Private Federated Learning on Medical Data},
  author={Malekzadeh, Mohammad and Hasircioglu, Burak and  Mital, Nitish and  Katarya, Kunal and Ozfatura,  Mehmet Emre and Gündüz, Deniz}, 
  journal= {The Second AAAI Workshop on Privacy-Preserving Artificial Intelligence (PPAI-21)},
  year={2021},
  url = {https://github.com/ipc-lab/private-ml-for-health}
}
```

## Collaboration/Contribution
We kindly welcome collaboration and/or contribution to this work. Please feel free to drop a line to us via email or by opening an issue.  
