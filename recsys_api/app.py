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
df.rename(columns={'location/city': 'location', 'index': 'idx'}, inplace=True)

df['deskripsi'] = df['deskripsi'].fillna('')
def cleaning(Text):
    Text = re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9,.]+',' ', Text)
    return Text
df['deskripsi'] = df['deskripsi'].apply(cleaning)

# define tf-pdf matrix
vectorizer_tfpdf = CountVectorizer()
term_freq_matrix = vectorizer_tfpdf.fit_transform(model['stemming'])

tf_matrix = normalize(term_freq_matrix, norm='l1', axis=1)

doc_frequency = (term_freq_matrix > 0).sum(axis=0)
doc_frequency = np.asarray(doc_frequency).flatten()

N = term_freq_matrix.shape[0]
pdf = np.log((N - doc_frequency + 0.5) / (doc_frequency + 0.5))

tf_pdf_matrix = tf_matrix.multiply(pdf)

from scipy.sparse import csr_matrix
tfpdf_matrix = csr_matrix(tf_pdf_matrix)

# construct cosine similarity score
cosine_sim = cosine_similarity(tfpdf_matrix, tfpdf_matrix)

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
        idx = df.index[df['idx'] == idx].tolist()[0]
        data_index = df.iloc[idx][['subkategori','location', 'nama', 'deskripsi']]
        data_index_json =data_index.to_dict()

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        top_indices = [i[0] for i in sim_scores]
        recommendation_data = df.iloc[top_indices][['subkategori','location', 'nama', 'deskripsi']]
        recommendation_data_json = recommendation_data.to_dict(orient='records')
        
        # Return JSON response
        return jsonify({
            "status_code": 200,
            "message": "success",
            "data_recommendation": recommendation_data_json,
            "data_index": data_index_json
        })

@app.route('/')
def home():
    page = int(request.args.get('page', 1))
    per_page = 20
    search_query = request.args.get('query', '')

    if search_query:
        filtered_data = df[df['nama'].str.contains(search_query, case=False, na=False)]
    else:
        filtered_data = df

    total = len(filtered_data)
    total_pages = (total // per_page) + (1 if total % per_page > 0 else 0)

    start = (page - 1) * per_page
    end = start + per_page
    paginated_data = filtered_data.iloc[start:end]
    # print(paginated_data)

    return render_template('index.html', 
                           companies=paginated_data.to_dict(orient='records'), 
                           page=page, 
                           total_pages=total_pages, 
                           query=search_query)

@app.route('/suggest', methods=['GET'])
def suggest():
    query = request.args.get('query')
    suggestions = df[df['nama'].str.contains(query, case=False, na=False)]
    return jsonify(list(suggestions['nama']))

@app.route('/company-detail-and-recommendation/<int:idx>')
def company_detail(idx):
    idx = df.index[df['idx'] == idx].tolist()[0]
    company = df.iloc[idx][['subkategori','location', 'nama', 'deskripsi']]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11] 
    top_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[top_indices].to_dict(orient='records')

    return render_template('company_detail_and_recommendation.html', 
                           company=company, 
                           recommendations=recommendations,
                           company_idx=df.iloc[idx][['idx']].tolist()[0])

@app.route('/more-recommendation/<int:idx>')
def load_more_companies(idx):
    idx = df.index[df['idx'] == idx].tolist()[0]
    page = int(request.args.get('page', 2))  # Default page is 2 if not specified
    per_page = 10  # Number of recommendations per page
    start_idx = ((page-1) * per_page) + 2
    end_idx = start_idx + per_page

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[start_idx:end_idx]
    top_indices = [i[0] for i in sim_scores]
    next_recommendations = df.iloc[top_indices].to_dict(orient='records')
    return jsonify({"recommendations": next_recommendations})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 8080)))