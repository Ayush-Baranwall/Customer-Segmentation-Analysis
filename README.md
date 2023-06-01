# CUSTOMER SEGMENTATION ANALYSIS USING CLUSTERING ALGORITHMS

This Jupyter Notebook demonstrates the application of different clustering algorithms on a dataset and evaluates their performance using various metrics. The notebook provides an overview of the code, explanations of the concepts used, and presents the results.
## Contents

    Introduction
    Dataset
    Clustering Algorithms
    Evaluation Metrics
    Code Execution
    Results
    License

## Introduction

Clustering is an unsupervised learning technique used to group similar data points together. It helps in discovering patterns and structures in the data without any predefined labels. This notebook showcases the application of four clustering algorithms: KMeans, Agglomerative Clustering, DBSCAN, and MeanShift. The algorithms are evaluated using three metrics: Silhouette Score, Calinski-Harabasz Score, and Davies-Bouldin Score.
## Dataset

The dataset contains information about customers of a supermarket mall. It includes the following information:

    Customer ID
    Gender
    Age
    Annual Income
    Spending Score

The spending score is a measure of how much money a customer spends at the mall. It is calculated based on the customer's purchase history.

The dataset can be used to segment customers and identify different types of customers. This information can be used to develop marketing campaigns that target specific customer segments.

Here are some examples of how the dataset can be used:

    Identify high-value customers who are likely to spend more money at the mall.
    Target customers with specific products or services based on their demographics and spending habits.
    Develop loyalty programs that encourage customers to spend more money at the mall.

The dataset is a valuable resource for businesses that want to better understand their customers and develop effective marketing strategies.
## The notebook utilizes the following clustering algorithms:

    KMeans: A centroid-based clustering algorithm that assigns each data point to the nearest centroid. The algorithm aims to minimize the within-cluster sum of squares.
    Agglomerative Clustering: A hierarchical clustering algorithm that starts with individual data points as clusters and progressively merges them based on a distance metric.
    DBSCAN: A density-based clustering algorithm that groups data points based on their density and connectivity. It can find clusters of arbitrary shapes and identify noise points.
    MeanShift: A density-based clustering algorithm that identifies cluster centers by iteratively shifting towards higher density regions. It does not require specifying the number of clusters in advance.

## Evaluation Metrics

The performance of the clustering algorithms is assessed using the following evaluation metrics:

    Silhouette Score: Measures how well each data point fits its assigned cluster compared to other clusters. The score ranges from -1 to 1, where higher values indicate better-defined clusters.
    Calinski-Harabasz Score: Calculates the ratio between the within-cluster dispersion and between-cluster dispersion. Higher scores indicate better-defined clusters.
    Davies-Bouldin Score: Measures the average similarity between clusters, where lower values indicate better-defined clusters.

## Code Execution

The code is executed in a Jupyter Notebook environment. To run the code:

    Ensure you have the necessary dependencies installed (see the Dependencies section).
    Place the "Mall_Customers.csv" dataset file in the same directory as the notebook.
    Execute each code cell in sequential order.

## Results

The code generates scatter plots for each clustering algorithm, displaying the data points with assigned cluster labels. It also prints the evaluation scores for each algorithm. Additionally, it determines the best clustering algorithm based on the Calinski-Harabasz score and displays the result.
Dependencies

The code relies on the following Python libraries:

    numpy
    pandas
    matplotlib
    scikit-learn

You can install these dependencies using pip:

bash

pip install numpy pandas matplotlib scikit-learn

## License

This project is licensed under the MIT License. See the LICENSE file for more information.

Dataset link - https://www.kaggle.com/datasets/shwetabh123/mall-customers
