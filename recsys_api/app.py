# import python libraries
from flask import Flask, jsonify, request, render_template, redirect, url_for, send_from_directory
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from flask_swagger_ui import get_swaggerui_blueprint
import os

app = Flask(__name__)
model = pd.read_csv('recsys_api/datamodel.csv')
df = pd.read_csv('recsys_api/dataset.csv')

df['deskripsi'] = df['deskripsi'].fillna('')
def cleaning(Text):
    Text = re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9,.]+',' ', Text)
    return Text
df['deskripsi'] = df['deskripsi'].apply(cleaning)

# define tf-pdf matrix
vectorizer_tfpdf = CountVectorizer()
term_freq_matrix = vectorizer_tfpdf.fit_transform(model['stemming'])

# Normalize the term frequency matrix to get TF
tf_matrix = normalize(term_freq_matrix, norm='l1', axis=1)

# Step 2: Compute Document Frequency (DF)
doc_frequency = (term_freq_matrix > 0).sum(axis=0)
doc_frequency = np.asarray(doc_frequency).flatten()

# Step 3: Compute Probabilistic Document Frequency (PDF)
N = term_freq_matrix.shape[0]
pdf = np.log((N - doc_frequency + 0.5) / (doc_frequency + 0.5))

# Step 4: Combine TF and PDF to get TF-PDF
tf_pdf_matrix = tf_matrix.multiply(pdf)

# Convert to sparse matrix format (similar to tfidf_vectors)
from scipy.sparse import csr_matrix
tfpdf_matrix = csr_matrix(tf_pdf_matrix)

# construct cosine similarity score
cosine_sim = cosine_similarity(tfpdf_matrix, tfpdf_matrix)

# define swagger
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Eventhings"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)

# localhost:8080/api/recommendation
@app.route('/api/recommendation', methods=['POST'])
def get_recommendations():
    if request.method == 'POST':
        idx = int(request.json['index'])
        data_index = df.iloc[idx][['kategori','subkategori','location/city', 'nama', 'deskripsi']]
        data_index_json =data_index.to_dict()

        # generate recommendations
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        top_indices = [i[0] for i in sim_scores]
        recommendation_data = df.iloc[top_indices][['kategori','subkategori', 'nama', 'location/city', 'deskripsi']]
        recommendation_data_json = recommendation_data.to_dict(orient='records')
        
        # Return JSON response
        return jsonify({
            "status_code": 200,
            "message": "success tfpdf",
            "data_recommendation": recommendation_data_json,
            "data_index": data_index_json
        })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 8080)))