# <snippet_imports>
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid
# </snippet_imports>

'''
Prerequisites:

1. Install the Custom Vision SDK. Run:
pip install --upgrade azure-cognitiveservices-vision-customvision

2. Create an "Images" folder in your working directory.

3. Download the images used by this sample from:
https://github.com/Azure-Samples/cognitive-services-sample-data-files/tree/master/CustomVision/ImageClassification/Images

This sample looks for images in the following paths:
<your working directory>/Images/Hemlock
<your working directory>/Images/Japanese_Cherry
<your working directory>/Images/Test
'''

# <snippet_creds>
# retrieve environment variables
ENDPOINT = 'https://cvux-prediction.cognitiveservices.azure.com/' #os.environ["VISION_TRAINING_ENDPOINT"]
prediction_key = os.environ["VISION_PREDICTION_KEY"]
prediction_resource_id = '/subscriptions/fdb5d841-c563-47cd-9530-53dbd266c1fb/resourceGroups/cv-ux-rg/providers/Microsoft.CognitiveServices/accounts/cvux-Prediction' #os.environ["VISION_PREDICTION_RESOURCE_ID"]

# </snippet_creds>

# <snippet_auth>
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
# </snippet_auth>

# <snippet_create>
publish_iteration_name = "detectModel"

# <snippet_upload>
base_image_location = os.path.join (os.path.dirname(__file__), "images-d")

project_id = "2f9a0d30-3773-4f43-9f81-874b77da08ed"

# <snippet_test>
# Now there is a trained endpoint that can be used to make a prediction
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

# <snippet_test>
# Now there is a trained endpoint that can be used to make a prediction

# Open the sample image and get back the prediction results.
with open(os.path.join (base_image_location, "test", "test_image.jpg"), mode="rb") as test_data:
    results = predictor.detect_image(project_id, publish_iteration_name, test_data)

# Display the results.    
for prediction in results.predictions:
    print("\t" + prediction.tag_name + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(prediction.probability * 100, prediction.bounding_box.left, prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))
# </snippet_test>

# <snippet_delete>
# You cannot delete a project with published iterations, so you must first unpublish them.
#print ("Unpublishing project...")
#trainer.unpublish_iteration(project.id, iteration.id)

#print ("Deleting project...")
#trainer.delete_project (project.id)
# </snippet_delete>
