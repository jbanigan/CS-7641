# CS7641 - Machine Learning: Unsupervised Learning.
Project code files are located at: https://github.com/jbanigan/CS-7641/tree/main/Unsupervised%20Learning
two clustering algorithms were analyzed with four dimensionality algorithms, and neural networks were applied post dimensionality reduction.
## Clustering Algorithms:
1. k-means
2. Expectation Maximization
## Dimensionality Reduction:
1. PCA
2. ICA
3. Randomized Projections
4. Factor Analysis
## Code:
All code was written in Jupyter Notebook
1. Clustering Algorithms.ipynb - includes both of the clustering algorithms that run on the datasets alone.
2. Neural Network.ipynb - contains the code to run the clustering algorithms and then the Neural Network and both datasets.
3. PCA.ipynb - this notebook contains the code to run PCA with both clustering algorithms and a Neural Network and both datasets.
4. ICA.ipynb - this notebook contains the code to run ICA with both clustering algorithms and a Neural Network and both datasets.
5. RP.ipynb - this notebook contains the code to run RP with both clustering algorithms and a Neural Network and both datasets.
6. FA.ipynb - this notebook contains the code to run FA with both clustering algorithms and a Neural Network and both datasets.

Libraries required to run the code:
scikit-learn, numpy, pandas, matplotlib
## Running the code:
1. Open the notebook in Jupyter Notebook, press Kernel tab > Restart & Run All
2. This will print all the required plots for the project.
3. The code only runs 1 dataset at a time, to switch between the dataset change the dataset_select integer the second cell of the notebook to 0 for wine dataset and 1 for cancer dataset
## Datasets:
1. Wine Quality Dataset, can be downloaded at: https://archive.ics.uci.edu/ml/datasets/wine+quality
2. Breast Cancer Wisconsin, can be downloaded at: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)
## Folders
1. **data** folder contains all datasets required for the project
2. **img** folder contains all images of all the plots produced by the code
