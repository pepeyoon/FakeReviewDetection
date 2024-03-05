### Fake Review Detection Model

## Project Description

This purpose of this project is to assist users in distinguishing between authentic and fake product reviews. 
We tackle this challenge using three primary classification models: Multinomial Naive Bayes, Random Forest Classifier, and Gradient Boosting Classifier.
Models Utilized
	1.	Multinomial Naive Bayes with TF-IDF: This technique is especially powerful when dealing with discrete data, as in text classification tasks. As part of the preprocessing step, we used TF-IDF vectorization on the review data.
	2.	Random Forest Classifier with TF-IDF and Word2Vec: We used a Random Forest model to enhance classification accuracy; it accomplishes this by generating numerous decision trees. Both TF-IDF and Word2Vec vectorization were experimented with during this process to assess their respective impacts on the model’s performance.
	3.	Gradient Boosting Classifier with Word2Vec: For this model, we selected Gradient Boosting due to its strong performance and adaptability. Word2Vec vectorization was useful in progressively refining the model and identifying patterns within the text data.
 
## Model Evaluation Metrics

The models’ effectiveness was assessed using the following evaluation metrics:
	•	Accuracy: This measures the overall correctness of the classification model.
	•	Sensitivity/Recall: This metric primarily evaluates the model’s capability to accurately identify genuine reviews. This is crucial as it’s preferable to prevent fake reviews from getting published.
	•	F1 Score: This provides a balanced overview of precision and sensitivity, thereby reflecting the model’s performance across these dimensions.
	•	AUROC: This metric measures the model’s ability to distinguish between true positive and true negative instances.
 
## Comparing Models

Each of the tested models demonstrated respectable performance across all measurements. However, the standout was the Random Forest model with TF-IDF vectorization - it consistently performed well across most metrics, apart from sensitivity where it was slightly surpassed by another model.

## API Implementation

In order to provide a user-friendly interface to the model, an API was implemented using the FastAPI package. When users provide a new review, the review is processed by the model; the returned output predicts whether the review is likely authentic. The API works by accepting user input, preprocessing the review, running it through the stored model, and returning the prediction to the user.
