from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the model using the Transformers library
pipe = pipeline("text-classification", model="elftsdmr/malware-url-detect")

@app.route('/classify-url', methods=['POST'])
def classify_url():
    try:
        # Get the URL from the POST request
        data = request.get_json()
        url = data['url']

        # Use the model to classify the URL
        result = pipe(url)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
