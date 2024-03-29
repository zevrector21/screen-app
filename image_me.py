# from transformers import ViTImageProcessor, ViTForImageClassification
# from PIL import Image
# import requests

# # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# # image = Image.open(requests.get(url, stream=True).raw)
# image = Image.open('files/bet1.jpg')

# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits
# # model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])



from transformers import pipeline
from PIL import Image
import requests

film_pipe = pipeline(
    "image-classification",
    model="pszemraj/beit-large-patch16-512-film-shot-classifier",
)

violation_pipe = pipeline(
    "image-classification",
    # model="AykeeSalazar/violation-classification-bantai_vit",
    model="philfriedo81/rare-politiker",
)

# url = "https://cdn-uploads.huggingface.co/production/uploads/60bccec062080d33f875cd0c/9YqYvv188ZccCMSzuv0KW.png"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open('files/samples/Screenshot from 2024-03-01 01-16-50.png')
result = violation_pipe(image)
print(result)



# from transformers import ViTImageProcessor, FlaxViTModel
# from PIL import Image
# import requests

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
# model = FlaxViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# inputs = processor(images=image, return_tensors="np")
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state
# import pdb
# pdb.set_trace()
# print('~~~~~', last_hidden_states)

