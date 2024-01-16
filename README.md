
# Machine Leraning with texts
**Author:** [@harshpx](https://github.com/harshpx)
## Overview
A Flask based web app that contains multiple Machine Learning models using NLP.
(UI is designed using Basic HTML and CSS)

**Currently this app contains:**
* Emotion Detector😊😮😱🥰😠
* Sentiment Detector👍🏻👎🏻
* Language Detector(عربي, हिंदी, Engilsh, แบบไทย etc.)

It uses various trained Machine Learning and Deep Learning models at the backend, and API is created using Flask.


#### To run this project on your local system: 

##### (Using git)
```
mkdir project
cd project
git clone https://github.com/harshpx/Language-and-Sentiment-Detector.git
pip install -r requirements.txt
app.py
```

## Brief Project Description
This Project currently uses 3 trained Machine Learning and Deep Learning Models.

1. **Emotion Detection**
* It is a Multiclass classification Model that uses various small datasets of texts and labelled emotion from Kaggle. 
* Data from various Sources are then combined to compile a big dataset (Data Augmentation).
* The overall combined dataset consists of approx 30,000 labelled texts, that spans accross 9 emotions. Namely: `anger`, `disgust`, `fear`, `guilt`, `joy`, `love`, `sadness`, `shame` and `surprise`. (A 9 class classification problem)
* Various Machine Learning classification Models are applied on the processed data.
* The best performing models were: `Linear SVM` and `Logistic Regression`, both achieved an accuray of ~*80%*.
* Hence a `Stacking Ensemble Classifier` is used to combine the above two models and it performed slightly better than both of them (as expected).
* Final accuracy achieved: **81%**. (Decent for a 9-class classification).

Read More about Stacking Ensemble: [Stacking Ensemble-Machine Learning Mastery](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)
and Support Vector Machines(SVM): [SVMs-Towards Data Science](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)
        

2. **Sentiment Detection**
* It is also a Multiclass classification Model that uses a modified dataset derived from `Sentiment 140`, that has 50k tweets spanning accross 3 sentiments. Namely: `Positive`, `Neutral` and `Negative`. (A 3 class classification problem)
* Various Machine Learning Classifiers like: `Logistic Regression`, `SVM`, `Decision Trees`, `Naive Bayes`, `KNN` etc. are tested during Hyperparameter Tuning (using `GridSearchCV()`)
`Logistic Regression` performed best among them and achieved an accuray of **82%**.

Read More about Logistic Regression: [Logistic Regression-Towards Data Science](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)
* Deep Learning Models are also applied on this dataset. A `Bi-Directional LSTM` based Neural Network is trained upon this dataset, which further enhanced the accuracy to **83%** (slight, but still relevant)
Read more about RNNs and LSTMs: [RNNs and LSTMs-Builtin.com](https://builtin.com/data-science/recurrent-neural-networks-and-lstm)


3. **Language Detection**
* It is a Multiclass classification Model, uses a Dataset of 22000 text samples of 22 different languages. (1000 data instances of each language)
* Model is trained using `Naive Bayes` algorithm. ```Naive Bayes is a classification algorithm based on Bayes' theorem. It assumes that the features in the input data are independent of each other given the class label.```

Read more about Naive Bayes: [Naive Bayes Classifier Explained-Analytics Vidya](https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/)


## Source Codes
* [Kaggle: Emotion-detection-from-texts](https://www.kaggle.com/code/harshpriye/emotion-detection-from-texts)
* [Kaggle: Sentiment-analysis-from-texts](https://www.kaggle.com/code/harshpriye/sentiment-analysis-from-texts)
* [Kaggle: Sentiment Analysis-Bi-LSTM](https://www.kaggle.com/code/harshpriye/sentiment-analysis-bi-lstm)
* [Kaggle: Language-Detection-from-Texts](https://www.kaggle.com/code/harshpriye/language-detection-from-texts)
