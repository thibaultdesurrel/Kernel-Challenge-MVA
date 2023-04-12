# Kaggle challenge for the course Kernel Methods - MVA

###### Alessandro Cecchini and Thibault de Surrel
[Link to the kaggle challenge](https://www.kaggle.com/competitions/data-challenge-kernel-methods-2022-2023/overview)

To run the training and the testing of the Weisfeiler-Lehman subtree kernel on the training and testing dataset, run
`python start_WL.py`
This script takes two arguments :
- `--h` the height of the Weisfeiler-Lehman subtree kernel (default : 2)
- `--C` the regularization parameter C of the SVM (default : 10)

The default values are the ones for which we obtained our best score (80.1 %) on the testing set.
