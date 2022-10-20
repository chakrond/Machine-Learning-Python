# Machine-Learning-Python

This repo contains the example work of the following:

Supervised Learning
  - Regression
    - Simple Linear Regression
    - Multiple Linear Regression
    - Polynomial Regression
    - Support Vector Regression (SVR)
    - Decision Tree Regression
    - Random Forest Regression
  
  Regression Comparison
    ![Regression](https://user-images.githubusercontent.com/94983485/179699151-0bc06c20-a9b5-44a6-883b-81d37a38957f.png)

  
  - Classification
    - Logistic Regression
    - K-Nearest Neighbors (K-NN)
    - Support Vector Machine (SVM)
    - Kernel SVM
    - Naive Bayes
    - Decision Tree Classification
    - Random Forest Classification
   
  Classification Comparison
    ![Classification](https://user-images.githubusercontent.com/94983485/179698487-c192e7de-bc5e-4013-ad61-f4c97add457a.png)
  
  Classification Decision Boundary Comparison
    ![Classification_Decission_Boundary](https://user-images.githubusercontent.com/94983485/179698520-6133db07-a41e-412f-ba6f-dcf99e498c63.png)


Unsupervised Learning
  - Clustering
    - K-Means Clustering
    - Hierarchical Clustering
 
  Clustering Comparison
    ![Clustering Comparison](https://user-images.githubusercontent.com/94983485/179699780-b048f7f0-7436-443f-ac03-3d202d43ccde.png)
  
  Optimal Comparison
    ![Optimal Comparison](https://user-images.githubusercontent.com/94983485/179699916-60d32f30-e637-4418-b276-520c08d1e9f3.png)


Association Rule Learning
  - Apriori
  - Eclat
  
  
Reinforcement Learning
  - Upper Confidence Bound (UCB)
  - Thompson Sampling
  
  UCB vs Thompson at iteration of 500
    ![UCB_vs_Thompson_500](https://user-images.githubusercontent.com/94983485/179701438-8eb03ee0-d899-495a-a9ed-c18a88a4d6ce.png)

  UCB vs Thompson at iteration of 10000
    ![UCB_vs_Thompson_1000](https://user-images.githubusercontent.com/94983485/179701546-4fb63e3d-d4cb-40b7-819c-7f6b99234bcb.png)


Natural Language Processing
  - Bag of words (CountVectorizer, Regression)
  
  CountVectorizer & Prediction(Regression)
    ![NLP_BagOfWords](https://user-images.githubusercontent.com/94983485/179868269-d480b1b6-8e3e-4c21-98fe-fea2a931f557.png)


Dimensionality Reduction
  - Supervised
    - Linear Discriminant Analysis (LDA)
    
  - Unsupervised
    - Principal Component Analysis (PCA)
    - Kernel PCA

  Decision Boundary Comparison (Training Set)
    ![decision_boundary_compare](https://user-images.githubusercontent.com/94983485/180927605-77add443-988f-4a1a-a472-7f0906f76ebf.png)

  Confusion Matrix Comparison & Accuracy Score (Test Set)
    ![confusion_mat_compare](https://user-images.githubusercontent.com/94983485/180927896-71294e2e-bb63-4aad-a217-f0527100a22c.png)


XGBoost
  - XGBoost Classifier


Model Validation
  - k-Fold Cross Validation
  - Grid Search


Deep Learning

  - Supervised
  
    - Artificial Neural Networks (ANN)
      - Binary Classification
        ![Figure_1](https://user-images.githubusercontent.com/94983485/184515674-73c0a63f-14e3-4d69-870e-8acda9b72c1c.png)
        
      - Model Performance Scores
        ![perf_scores](https://user-images.githubusercontent.com/94983485/193439657-23a7c500-e5d6-4462-8391-4a8d5a25525e.png)

      - Multiple Classification - Experiments over learning rate parameters
        ![multiple_classification_softmax](https://user-images.githubusercontent.com/94983485/184528599-ef56881f-8df3-466d-9888-53fcd06dd80f.png)

      - Overall Classification Performance (Softmax) - Experiments over learning rate parameters
        ![classification_performance](https://user-images.githubusercontent.com/94983485/184553831-55f606e8-38a1-44f3-a344-2dd736ac567a.png)
        
      - Experiments over hidden layers parameters
        ![Exprms_hLayers](https://user-images.githubusercontent.com/94983485/184788667-0387c8f5-16b9-4efd-beb6-2a13b1ee153e.JPG)
        
      - Experiments over number of units per hidden layer parameters
        ![hLayers_nUnits](https://user-images.githubusercontent.com/94983485/185781330-bced05a2-ae3d-458c-bafb-fb6716bcc769.png)
        
      - Overall Classification Accuracy
        ![overall_performance_category](https://user-images.githubusercontent.com/94983485/190528818-e35d8049-4acf-4e8e-8b55-5b1fd5c09270.JPG)
        
      - Data Inspection
        ![data inspection](https://user-images.githubusercontent.com/94983485/190287084-a63c3d4a-f798-4e9d-b2fd-d1d884ef08f0.JPG)
      
      - Result Correlation
        ![results_correlation](https://user-images.githubusercontent.com/94983485/190287653-903f1612-e43d-4b9c-9570-b4d835c35aca.JPG)
        
    - Fully-connected Feedforward Neural Network (FFN)
      - MNIST Image Data
        ![MNIST_image](https://user-images.githubusercontent.com/94983485/192119930-4b5419c6-4c8a-45d3-9a99-b3180f7516d2.png)
      
      - Data Inspection
        ![data_inspection](https://user-images.githubusercontent.com/94983485/192686869-9a1a7201-f2c0-407c-b88a-5ccd5fcd6323.png)

      - Model's Weights Distribution
        ![weights_distribution](https://user-images.githubusercontent.com/94983485/192119967-0cb5fc34-925f-4b87-b92e-8c140df6b0cb.png)
      
      - Model's Performance Scores
        ![perf_scores_mnist](https://user-images.githubusercontent.com/94983485/193783987-0c7b4b9f-4cee-42a6-b100-54fe92b9eae0.png)


    - Convolutional Neural Networks (CNN)
    - Recurrent Neural Networks (RNN)
    
  - Unsupervised
    - Self Organizing Maps
    - Boltzmann Machines
    - AutoEncoders
      - Denoised
        ![AE_Denoised](https://user-images.githubusercontent.com/94983485/196858508-87b050f2-f524-48e6-8ade-fe057c06472a.png)
      
      - Denoised from occlusion
        ![AE_Denoised_Occlusion](https://user-images.githubusercontent.com/94983485/196858637-962c4ccc-a9f8-4c8d-9723-da4fc5d21847.png)
      
      - Occlusion Effect
        ![Effect of occlusion](https://user-images.githubusercontent.com/94983485/196858693-a06a8a97-a672-4636-bb74-c4bc8de70dc4.JPG)


    
  - Function
    - Sigmoid
      ![sigmoid_relu](https://user-images.githubusercontent.com/94983485/183615058-fe96a2af-28d6-44c9-b7bd-e3ca4f7e6efe.jpg)

    - Softmax
      ![softmax](https://user-images.githubusercontent.com/94983485/183615232-d8dd46f6-500d-4cac-b567-292d3ec005f6.png)

    - Mean Sample
      ![hist_sampling](https://user-images.githubusercontent.com/94983485/183615372-9bcc96a5-b875-4b81-b953-be566d9855f9.png)  
    
    - Gradient Descent 1D Animated
      ![gradient_descent1D_animated](https://user-images.githubusercontent.com/94983485/183615587-9094fb34-aaad-46c1-9120-cabe5a33e581.png)
    
    - Gradient Descent 2D Animated
    
      ![gradient_descent_2D](https://user-images.githubusercontent.com/94983485/184464016-c1c72cf6-6b6f-4e33-ad1a-0ee1c9b0119b.png)
      
      
      
      
      
      
      
