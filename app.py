from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("FredZhang7/malware-phisher")
model = AutoModelForSequenceClassification.from_pretrained("FredZhang7/malware-phisher")

app = Flask(__name__)

# Load the model using the Transformers library
def classify_url_with_scores(url):
    # Tokenize the URL and ensure it has a maximum length (adjust this as needed)
    inputs = tokenizer(url, truncation=True, padding=True, max_length=64, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted probabilities (scores) for each class
    probabilities = torch.softmax(outputs.logits, dim=1).tolist()[0]

    # Get the predicted class (0 for benign, 1 for malicious)
    prediction = torch.argmax(outputs.logits, dim=1).item()

    benign_score = probabilities[1] * 100
    malicious_score = probabilities[0] * 100

    return {
        "Benign URL": benign_score,
        "Malicious URL": malicious_score,
        "Prediction": "Benign URL" if prediction == 1 else "Malicious URL"
    }


@app.route('/classify-url', methods=['POST'])
def classify_url():
    try:
        # Get the URL from the POST request
        data = request.get_json()
        url = data['url']

        # Use the model to classify the URL
        result = classify_url_with_scores(url)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
