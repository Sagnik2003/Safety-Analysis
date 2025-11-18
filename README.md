
# Safety Analysis using Machine Learning

## Overview
**Safety Analysis** is a machine learning-based system designed to evaluate the safety of roads based on street images.  
This project is trained on images collected throughout the **IIT Bhubaneswar campus**, captured under diverse environmental and lighting conditions.

The system analyzes various visual and contextual features from street images — such as **human activity**, **time of day**, **presence of vehicles**, **illumination**, and **surrounding infrastructure** — to classify areas into one of three safety categories:
- 🟢 **Safe**
- 🟡 **Risky**
- 🔴 **Dangerous**

---

## Approach

### 1. **Data Collection**
Street images were captured across multiple campus locations at different times of the day and under varying weather conditions to ensure diversity.

Our dataset is openly available on [kaggle/Road Safety Analysis - IIT Bhubaneswar](https://www.kaggle.com/datasets/sagnikkayalcse52/road-safety-analysis-iit-bhubaneswar)

### 2. **Feature Extraction**
We detected and manually labeled features such as:
- Human activities  
- Vehicles  
- Lighting conditions  
- Nearby monuments/infrastructure  
- Time of day
- Nearby Vegetation 

These were annotated in a CSV file along with the safety class label.

### 3. **Image Preprocessing**
To obtain compact and meaningful image representations, the following preprocessing steps were performed:
- Image compression and resizing  
- Multiple convolutional passes using **manual kernels** (Sobel, Laplacian, DoG, etc.)  
- Pooling operations for dimensionality reduction  
- Flattening to a **1D feature vector**

### 4. **Feature Fusion**
The **image-derived vector** was concatenated with the **CSV-based contextual features** to form the complete input dataset (**X**).  
This combined representation captures both **visual** and **semantic** safety cues.

### 5. **Model Training**
Multiple machine learning models were trained and compared to identify the most effective classifier for road safety prediction.  
Typical models explored include:
- Logistic Regression
- QDA
- Naive Bayes
- KNN Classifier
- Decison Trees 
- Random Forest  
- Support Vector Machine (SVM)  
- Gradient Boosting  

The final output predicts whether a given street environment is **Safe**, **Risky**, or **Dangerous**.

---

## Future Work
This system will be **scaled up** to handle large-scale datasets and generalized to analyze streets across **any city or region**.  
Further improvements will include:
- Deep feature extraction using CNNs  
- Integration with GPS and temporal data  
- Real-time safety analysis via mobile or drone-based applications

---

## Repository Structure
```

Safety-Analysis/
│
├── Features
│   ├── Contextual Features    
│   ├── Image Features
│
├── Notebooks/
│   ├── Scripts                # Image preprocessing and feature extraction
│   ├── models.ipynb           # Various Models used
│
├── results/
│   ├── evaluation_metrics.csv # Model performance metrics
│   ├── confusion_matrix       # Visualization of model results
│
└── README.md

```

---

## Tech Stack
- **Python** (NumPy, Pandas, Scikit-learn, OpenCV, Matplotlib)
- **Jupyter Notebook**
- **Manual Feature Engineering + Machine Learning Models**

---

## Contributors
- **Uppara Kadiyala Nitin Krishna**  
  Ph.D, Computer Science and Engineering  
  *Indian Institute of Technology Bhubaneswar*
- **Sagnik Kayal**  
  M.Tech, Artificial Intelligence  
  *Indian Institute of Technology Bhubaneswar*
- **Chelluboina Raj Pavan**  
  M.Tech, Robotics and Artificial Intelligence  
  *Indian Institute of Technology Bhubaneswar*
- **Krutarth Jankat**  
  B.Tech, Electrical Engineering  
  *Indian Institute of Technology Bhubaneswar*

---

## Acknowledgment
This project is part of the ML course project at **IIT Bhubaneswar**, aiming to create data-driven tools for improving **road and pedestrian safety** through intelligent visual analysis.
```
