from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Initialize Flask App
app = Flask(__name__)
app.secret_key = "careerpath-ai-secret-key"  # REQUIRED


# Load Data
data = pd.read_csv("Datasets/job-skill-data.csv")
data['Skills'] = data['Skills'].fillna('').astype(str)
data['Job Title'] = data['Job Title'].fillna('').astype(str)

# Ensure optional columns exist
for col in ['Job Description', 'Certifications']:
    if col not in data.columns:
        data[col] = "N/A"

# Load or train TF-IDF model

# -------- Skills Vectorizer -------- #

SKILL_VECTORIZER_PATH = os.path.join("models", "skill_vectorizer.pkl")
SKILL_VECTOR_PATH = os.path.join("models", "skill_vectors.pkl")

if os.path.exists(SKILL_VECTORIZER_PATH) and os.path.exists(SKILL_VECTOR_PATH):
    with open(SKILL_VECTORIZER_PATH, "rb") as f:
        skill_vectorizer = pickle.load(f)
    with open(SKILL_VECTOR_PATH, "rb") as f:
        skill_vectors = pickle.load(f)
else:
    skill_vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(','), lowercase=True, token_pattern=None)
    skill_vectors = skill_vectorizer.fit_transform(data['Skills'])
    with open(SKILL_VECTORIZER_PATH, "wb") as f:
        pickle.dump(skill_vectorizer, f)
    with open(SKILL_VECTOR_PATH, "wb") as f:
        pickle.dump(skill_vectors, f)

# -------- Job Title Vectorizer -------- #

JOB_VECTORIZER_PATH = os.path.join("models", "job_title_vectorizer.pkl")
JOB_VECTOR_PATH = os.path.join("models", "job_title_vectors.pkl")

if os.path.exists(JOB_VECTORIZER_PATH) and os.path.exists(JOB_VECTOR_PATH):
    with open(JOB_VECTORIZER_PATH, "rb") as f:
        job_title_vectorizer = pickle.load(f)
    with open(JOB_VECTOR_PATH, "rb") as f:
        job_title_vectors = pickle.load(f)
else:
    job_title_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    job_title_vectors = job_title_vectorizer.fit_transform(data['Job Title'])
    with open(JOB_VECTORIZER_PATH, "wb") as f:
        pickle.dump(job_title_vectorizer, f)
    with open(JOB_VECTOR_PATH, "wb") as f:
        pickle.dump(job_title_vectors, f)

# Recommendation Function
def recommend_jobs(user_input, top_n=10):
    user_input_clean = ', '.join([s.strip().lower() for s in user_input.split(',') if s.strip()])
    user_vector = skill_vectorizer.transform([user_input_clean])
    similarity = cosine_similarity(user_vector, skill_vectors).flatten()

    # Work on a copy to avoid modifying global data
    temp_data = data.copy()
    temp_data['Match_%'] = (similarity * 100).round(2)

    def find_missing(job_skills, user_skills):
        job_skills_list = [s.strip() for s in str(job_skills).lower().split(',') if s.strip()]
        missing = [s for s in job_skills_list if s.lower() not in user_skills]
        return ', '.join(missing) if missing else 'None'

    temp_data['Missing_Skills'] = temp_data['Skills'].apply(lambda x: find_missing(x, user_input_clean))
    return temp_data.sort_values(by='Match_%', ascending=False).head(top_n)


def recommend_skills(job_title, top_n=5):
    job_vector = job_title_vectorizer.transform([job_title.lower()])
    similarity = cosine_similarity(job_vector, job_title_vectors).flatten()

    temp = data.copy()
    temp['Match_%'] = (similarity * 100).round(2)

    top_jobs = temp.sort_values(by='Match_%', ascending=False).head(top_n)

    # collect skills
    skills_set = set()
    for skills in top_jobs['Skills']:
        for skill in skills.split(','):
            skills_set.add(skill.strip())

    return {
        "Job Title": job_title,
        "Recommended Skills": sorted(skills_set),
        "Top Matching Jobs": top_jobs[
            [
                "Job Title",
                "Match_%",
                "Job Description",
                "Skills",
                "Certifications"
            ]
        ]
    }
# ---------------- ROUTES ---------------- #

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/skills')
def skills():
    return render_template('skills.html')

@app.route('/jobs')
def jobs():
    return render_template('jobs.html')

# @app.route('/jobs', methods=['GET', 'POST'])
# def jobs():
#     results = None
#     if request.method == 'POST':
#         job_input = request.form['job']
#         results = data[data['Job Title'].str.contains(job_input, case=False, na=False)]
#     return render_template('jobs.html', results=results)

@app.route('/dashboard')
def dashboard():
    todo = session.get('todo', [])
    completed = session.get('completed', [])
    target_jobs = session.get('target_jobs', [])
    
    total_skills = len(todo) + len(completed)
    skills_acquired = len(completed)
    missing_skills = len(todo)
    
    return render_template('dashboard.html',
        todo=todo,
        completed=completed,
        target_jobs=target_jobs,
        total_skills=total_skills,
        skills_acquired=skills_acquired,
        missing_skills=missing_skills,
        total_jobs=len(target_jobs)
    )

@app.route('/mark-complete', methods=['POST'])
def mark_complete():
    req_data = request.get_json()
    skill = req_data.get('skill')
    
    if not skill:
        return jsonify({'status': 'error'}), 400
    
    if 'completed' not in session:
        session['completed'] = []
    
    todo = session.get('todo', [])
    if skill in todo:
        todo.remove(skill)
        session['todo'] = todo
    
    if skill not in session['completed']:
        session['completed'].append(skill)
    
    session.modified = True
    return jsonify({
        'status': 'success',
        'todo': session['todo'],
        'completed': session['completed']
    })

@app.route('/mark-incomplete', methods=['POST'])
def mark_incomplete():
    req_data = request.get_json()
    skill = req_data.get('skill')
    
    if not skill:
        return jsonify({'status': 'error'}), 400

    if 'todo' not in session:
        session['todo'] = []

    completed = session.get('completed', [])
    if skill in completed:
        completed.remove(skill)
        session['completed'] = completed

    if skill not in session['todo']:
        session['todo'].append(skill)

    session.modified = True
    return jsonify({
        'status': 'success',
        'todo': session['todo'],
        'completed': session['completed']
    })

@app.route('/remove-skill', methods=['POST'])
def remove_skill():
    req_data = request.get_json()
    skill = req_data.get('skill')
    
    todo = session.get('todo', [])
    completed = session.get('completed', [])
    
    if skill in todo:
        todo.remove(skill)
        session['todo'] = todo
    if skill in completed:
        completed.remove(skill)
        session['completed'] = completed
        
    session.modified = True
    return jsonify({'status': 'success'})

@app.route('/remove-target-job', methods=['POST'])
def remove_target_job():
    req_data = request.get_json()
    job = req_data.get('job_title')
    
    target_jobs = session.get('target_jobs', [])
    if job in target_jobs:
        target_jobs.remove(job)
        session['target_jobs'] = target_jobs
    
    session.modified = True
    return jsonify({'status': 'success'})

@app.route('/get-session-data')
def get_session_data():
    todo = session.get('todo', [])
    completed = session.get('completed', [])
    target_jobs = session.get('target_jobs', [])
    total = len(todo) + len(completed)
    percent = round((len(completed) / total * 100), 1) if total > 0 else 0
    
    return jsonify({
        'todo': todo,
        'completed': completed,
        'target_jobs': target_jobs,
        'percent': percent
    })

@app.route('/about')
def about():
    return render_template('about.html')

# API endpoint for live recommendations
@app.route('/recommend_jobs', methods=['POST'])
def recommend_jobs_api():
    data_json = request.get_json()
    user_skills = data_json.get('skills', '').strip()

    if not user_skills:
        return jsonify([])

    results = recommend_jobs(user_skills, top_n=10)
    user_skills_list = [s.strip() for s in user_skills.split(',') if s.strip()]
    display_user_skills = ', '.join(user_skills_list)

    response = []
    for _, row in results.iterrows():
        response.append({
            "Job Title": row.get("Job Title", ""),
            "Job Description": row.get("Job Description", "N/A"),
            "Match_%": row.get("Match_%", 0),
            "Skills": row.get("Skills", ""),
            "Your Skills": display_user_skills,
            "Missing_Skills": row.get("Missing_Skills", ""),
            "Certifications": row.get("Certifications", "N/A"),
        })

    return jsonify(response)

@app.route('/recommend_skills', methods=['POST'])
def recommend_skills_api():
    data_json = request.get_json()
    job_title = data_json.get('job_title', '').strip()

    if not job_title:
        return jsonify({})

    result = recommend_skills(job_title, top_n=5)

    response = {
        "Job Title": result["Job Title"],
        "Recommended Skills": result["Recommended Skills"],
        "Top Matching Jobs": result["Top Matching Jobs"].to_dict(orient='records')
    }

    return jsonify(response)

@app.route("/add-to-todo", methods=["POST"])
def add_to_todo():
    data_json = request.get_json()
    skill = data_json.get("skill")

    if not skill:
        return jsonify({"status": "error", "message": "No skill received"}), 400

    if "todo" not in session:
        session["todo"] = []

    if skill not in session["todo"]:
        session["todo"].append(skill)

    session.modified = True

    return jsonify({
        "status": "success",
        "todo": session["todo"]
    })

@app.route("/add-target-job", methods=["POST"])
def add_target_job():
    data_json = request.get_json()
    job_title = data_json.get("job_title")

    if not job_title:
        return jsonify({"status": "error", "message": "No job received"}), 400

    if "target_jobs" not in session:
        session["target_jobs"] = []

    if job_title not in session["target_jobs"]:
        session["target_jobs"].append(job_title)

    session.modified = True

    return jsonify({
        "status": "success",
        "target_jobs": session["target_jobs"]
    })

# ---------------- RUN APP ---------------- #
if __name__ == '__main__':
    app.run(debug=True)