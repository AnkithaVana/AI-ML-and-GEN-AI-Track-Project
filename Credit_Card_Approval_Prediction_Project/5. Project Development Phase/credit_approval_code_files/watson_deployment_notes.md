# IBM Watson Machine Learning Deployment Notes

1. Create an IBM Cloud account and open Watson Studio.
2. Create a project and associate a Watson Machine Learning service.
3. Upload `models/credit_card_model.joblib` as the model artifact.
4. Create an online deployment and copy the scoring endpoint.
5. Send applicant feature values to the endpoint using the same feature names used by the Flask app.

The Flask app is ready for local predictions. Watson deployment requires IBM Cloud credentials and service details, so those values are intentionally not hard-coded in this project.
