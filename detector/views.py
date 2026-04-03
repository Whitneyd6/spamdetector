from django.shortcuts import render

# Create your views here.
import os
import pickle
from django.conf import settings
from django.shortcuts import render

MODEL_PATH      = os.path.join(settings.BASE_DIR, 'detector/spam_model.pkl')
VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'detector/vectorizer.pkl')

# reading the models

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

# function to call the models to make predictions

def classify_message(request):
    result = None
    email_text = ""

    if request.method == 'POST':
        email_text = request.POST.get('email_text', '').strip()
        if email_text:
            msg_vec = vectorizer.transform([email_text])
            result  = model.predict(msg_vec)[0]

    return render(request, 'classify.html', {
        'result': result,
        'email_text': email_text
    })
