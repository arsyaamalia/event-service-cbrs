# import python libraries
from flask import Flask, jsonify, request, render_template, redirect, url_for, send_from_directory
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_swagger_ui import get_swaggerui_blueprint
import os

app = Flask(__name__)
model = pd.read_csv('datamodel.csv')
df = pd.read_csv('dataset.csv')

# define tf-abs matrix
count_vectorizer = CountVectorizer()
term_counts = count_vectorizer.fit_transform(model['stemming'])
term_counts_array = term_counts.toarray()
abs_freq = np.sum(term_counts_array, axis=0)
tf_abs = term_counts_array * abs_freq
from scipy.sparse import csr_matrix
tfabs_matrix = csr_matrix(tf_abs)

# construct cosine similarity score
cosine_sim = cosine_similarity(tfabs_matrix, tfabs_matrix)

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
        data_index = df.iloc[idx][['subkategori','location/city', 'nama', 'deskripsi']]
        data_index_json =data_index.to_dict()

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        top_indices = [i[0] for i in sim_scores]
        recommendation_data = df.iloc[top_indices][['subkategori', 'nama', 'location/city', 'deskripsi']]
        recommendation_data_json = recommendation_data.to_dict(orient='records')
        
        # Return JSON response
        return jsonify({
            "status_code": 200,
            "message": "success",
            "data_recommendation": recommendation_data_json,
            "data_index": data_index_json
        })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 8080)))