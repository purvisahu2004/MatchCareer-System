from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import os
import html
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import json
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'Unzila_MatchCareer')

# Properly encode database password to handle special characters
db_password = quote_plus(os.getenv('DB_PASSWORD', ''))

# Try MySQL first, fallback to SQLite for development
try:
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'matchcareer.db')}"
    # Test the connection
    from sqlalchemy import create_engine, text
    engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("Connected to SQLlite successfully!")
except Exception as e:
    print(f"MySQL connection failed: {e}")
    print("Falling back to SQLite for development...")
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'matchcareer.db')}"

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    # User progress data
    todo_skills = db.Column(db.Text, default='[]')  # JSON string
    completed_skills = db.Column(db.Text, default='[]')  # JSON string
    target_jobs = db.Column(db.Text, default='[]')  # JSON string

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_todo_skills(self):
        return json.loads(self.todo_skills) if self.todo_skills else []

    def set_todo_skills(self, skills):
        self.todo_skills = json.dumps(skills)

    def get_completed_skills(self):
        return json.loads(self.completed_skills) if self.completed_skills else []

    def set_completed_skills(self, skills):
        self.completed_skills = json.dumps(skills)

    def get_target_jobs(self):
        return json.loads(self.target_jobs) if self.target_jobs else []

    def set_target_jobs(self, jobs):
        self.target_jobs = json.dumps(jobs)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


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

# ...existing code...

# ...existing code...

def recommend_skills(job_title, top_n=10):
    job_vector = job_title_vectorizer.transform([job_title.lower()])
    similarity = cosine_similarity(job_vector, job_title_vectors).flatten()

    temp = data.copy()
    temp['Match_%'] = (similarity * 100).round(2)

    top_jobs = temp.sort_values(by='Match_%', ascending=False).head(top_n)

    # Check if the highest match is below a threshold (e.g., 20%)
    if top_jobs['Match_%'].max() < 20:
        return {"message": "I don't have data about this job title."}

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

# ...existing code...

# ...existing code...


def normalize_session_skill_items(items):
    normalized = []
    for item in items or []:
        if isinstance(item, str):
            normalized.append({"skill": item, "jobs": []})
        elif isinstance(item, dict):
            skill_name = item.get("skill", "") or ""
            jobs = item.get("jobs", [])
            if isinstance(jobs, str):
                jobs = [jobs]
            elif jobs is None:
                jobs = []
            elif not isinstance(jobs, list):
                jobs = list(jobs)
            normalized.append({"skill": skill_name, "jobs": jobs})
    return normalized


@app.before_request
def load_user_data():
    if current_user.is_authenticated:
        # User data is already loaded via the User model
        pass

# ---------------- AUTHENTICATION ROUTES ---------------- #

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return redirect(url_for('register'))

        user = User(username=username, email=email)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))

        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# ---------------- MAIN ROUTES ---------------- #

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/skills')
def skills():
    return render_template('skills.html')

@app.route('/jobs')
def jobs():
    return render_template('jobs.html')

@app.route('/dashboard')
@login_required
def dashboard():
    todo = current_user.get_todo_skills()
    completed = current_user.get_completed_skills()
    target_jobs = current_user.get_target_jobs()

    total_skills = len(todo) + len(completed)
    skills_acquired = len(completed)
    missing_skills = len(todo)
    completed_skill_names = [item['skill'] for item in completed]

    # -------- Helper Function -------- #
    def get_job_skill_match(job_title, completed_skill_names):
        job_row = data[data['Job Title'].str.lower() == job_title.lower()]

        if job_row.empty:
            return 0

        job_skills = job_row.iloc[0]['Skills']
        job_skills_list = [s.strip().lower() for s in job_skills.split(',') if s.strip()]

        if not job_skills_list:
            return 0

        completed_lower = [c.lower() for c in completed_skill_names]
        matched = [s for s in job_skills_list if s in completed_lower]

        return round(len(matched) / len(job_skills_list) * 100, 1)

    # -------- 1. PIE CHART (Skill Completion) -------- #
    pie_fig = go.Figure(data=[go.Pie(
        labels=['Completed', 'Remaining'],
        values=[skills_acquired, max(missing_skills, 1)],
        marker=dict(colors=['#48bb78', '#4a5568']),
        textinfo='label+percent',
        textfont=dict(color='#e2e8f0', size=12)
    )])

    pie_fig.update_layout(
        title="Skill Completion",
        title_font=dict(color='#e2e8f0', size=16),
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0')
    )

    pie_chart = pie_fig.to_html(full_html=False, include_plotlyjs=False)

    # -------- 2. HORIZONTAL BAR (Skill Progress per Job) -------- #
    if target_jobs:
        job_labels = []
        job_progress = []

        for job in target_jobs:
            progress = get_job_skill_match(job, completed_skill_names)
            job_labels.append(job)
            job_progress.append(progress)

        hbar_fig = go.Figure(go.Bar(
            x=job_progress,
            y=job_labels,
            orientation='h',
            marker=dict(color='#764ba2'),
            text=[f"{p}%" for p in job_progress],
            textposition='auto',
            textfont=dict(color='#e2e8f0')
        ))

        hbar_fig.update_layout(
            title="Skill Progress per Target Job",
            title_font=dict(color='#e2e8f0', size=16),
            xaxis=dict(title='Completion %', range=[0, 100], title_font=dict(color='#e2e8f0'), tickfont=dict(color='#e2e8f0')),
            yaxis=dict(tickfont=dict(color='#e2e8f0')),
            height=max(300, len(target_jobs) * 60),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0')
        )

        hbar_chart = hbar_fig.to_html(full_html=False, include_plotlyjs=False)
    else:
        hbar_chart = None

    # -------- 3. JOB MATCH BAR CHART -------- #
    if target_jobs:
        job_names = []
        job_match_scores = []

        for job in target_jobs:
            score = get_job_skill_match(job, completed_skill_names)
            job_names.append(job)
            job_match_scores.append(score)

        match_fig = go.Figure(go.Bar(
            x=job_names,
            y=job_match_scores,
            marker=dict(color='#48bb78'),
            text=[f"{s}%" for s in job_match_scores],
            textposition='auto',
            textfont=dict(color='#e2e8f0')
        ))

        match_fig.update_layout(
            title="Job Match Based on Your Skills",
            title_font=dict(color='#e2e8f0', size=16),
            yaxis=dict(title='Match %', range=[0, 100], title_font=dict(color='#e2e8f0'), tickfont=dict(color='#e2e8f0')),
            xaxis=dict(title='Jobs', title_font=dict(color='#e2e8f0'), tickfont=dict(color='#e2e8f0')),
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        match_chart = match_fig.to_html(full_html=False, include_plotlyjs=False)
    else:
        match_chart = None

    return render_template(
        'dashboard.html',
        pie_chart=pie_chart,
        hbar_chart=hbar_chart,
        match_chart=match_chart,

        # REQUIRED FOR STATS ROW ✅
        total_skills=total_skills,
        skills_acquired=skills_acquired,
        missing_skills=missing_skills,
        total_jobs=len(target_jobs),

        # existing
        todo=todo,
        completed=completed,
        target_jobs=target_jobs
    )

@app.route('/mark-complete', methods=['POST'])
@login_required
def mark_complete():
    req = request.get_json()
    skill = html.unescape(req.get('skill'))

    todo = current_user.get_todo_skills()
    completed = current_user.get_completed_skills()

    if skill:
        match = next((item for item in todo if item['skill'].strip().lower() == skill.strip().lower()), None)
        if match:
            todo = [item for item in todo if item['skill'].strip().lower() != skill.strip().lower()]
            existing = next((item for item in completed if item['skill'].strip().lower() == skill.strip().lower()), None)
            if existing:
                for job in match.get('jobs', []):
                    if job not in existing['jobs']:
                        existing['jobs'].append(job)
            else:
                completed.append({
                    'skill': match['skill'],
                    'jobs': match.get('jobs', [])
                })

    current_user.set_todo_skills(todo)
    current_user.set_completed_skills(completed)
    db.session.commit()

    return jsonify({
        "status": "success",
        "todo": todo,
        "completed": completed
    })

@app.route('/mark-incomplete', methods=['POST'])
@login_required
def mark_incomplete():
    req = request.get_json()
    skill = html.unescape(req.get('skill'))

    todo = current_user.get_todo_skills()
    completed = current_user.get_completed_skills()

    if skill:
        match = next((item for item in completed if item['skill'].strip().lower() == skill.strip().lower()), None)
        if match:
            completed = [item for item in completed if item['skill'].strip().lower() != skill.strip().lower()]
            existing = next((item for item in todo if item['skill'].strip().lower() == skill.strip().lower()), None)
            if existing:
                for job in match.get('jobs', []):
                    if job not in existing['jobs']:
                        existing['jobs'].append(job)
            else:
                todo.append({
                    'skill': match['skill'],
                    'jobs': match.get('jobs', [])
                })

    current_user.set_todo_skills(todo)
    current_user.set_completed_skills(completed)
    db.session.commit()

    return jsonify({
        "status": "success",
        "todo": todo,
        "completed": completed
    })

@app.route('/remove-skill', methods=['POST'])
@login_required
def remove_skill():
    req_data = request.get_json()
    skill = html.unescape(req_data.get('skill'))

    todo = current_user.get_todo_skills()
    completed = current_user.get_completed_skills()

    if skill:
        todo = [item for item in todo if item['skill'].strip().lower() != skill.strip().lower()]
        completed = [item for item in completed if item['skill'].strip().lower() != skill.strip().lower()]
        current_user.set_todo_skills(todo)
        current_user.set_completed_skills(completed)
        db.session.commit()

    return jsonify({
        'status': 'success',
        'todo': todo,
        'completed': completed
    })

@app.route('/remove-target-job', methods=['POST'])
@login_required
def remove_target_job():
    req_data = request.get_json()
    job = html.unescape(req_data.get('job_title'))

    target_jobs = current_user.get_target_jobs()
    todo = current_user.get_todo_skills()

    if job in target_jobs:
        target_jobs.remove(job)
        current_user.set_target_jobs(target_jobs)

    # Remove job from skill associations and delete skills with no jobs left
    updated_todo = []
    for item in todo:
        if job in item.get('jobs', []):
            item['jobs'] = [j for j in item.get('jobs', []) if j != job]
        # Only keep the skill if it still has associated jobs
        if item.get('jobs'):
            updated_todo.append(item)

    current_user.set_todo_skills(updated_todo)
    db.session.commit()
    return jsonify({"status": "success", "target_jobs": target_jobs, "todo": updated_todo})

@app.route('/get-session-data')
@login_required
def get_session_data():
    todo = current_user.get_todo_skills()
    completed = current_user.get_completed_skills()
    target_jobs = current_user.get_target_jobs()
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

# ...existing code...

@app.route('/recommend_skills', methods=['POST'])
def recommend_skills_api():
    data_json = request.get_json()
    job_title = data_json.get('job_title', '').strip()
    top_n = int(data_json.get('top_n', 10))  # ← read from request

    if not job_title:
        return jsonify({})

    result = recommend_skills(job_title, top_n=top_n)

    if "message" in result:
        return jsonify({"message": result["message"]})

    response = {
        "Job Title": result["Job Title"],
        "Recommended Skills": result["Recommended Skills"],
        "Top Matching Jobs": result["Top Matching Jobs"].to_dict(orient='records')
    }

    return jsonify(response)

# ...existing code...

@app.route('/autocomplete-skills')
def autocomplete_skills():
    query = request.args.get('q', '').strip().lower()
    if not query:
        return jsonify([])
    
    all_skills = set()
    for skills_str in data['Skills']:
        for skill in skills_str.split(','):
            skill = skill.strip()
            if skill:
                all_skills.add(skill)
    
    matches = [s for s in all_skills if query in s.lower()]
    return jsonify(sorted(matches)[:10])


@app.route('/autocomplete-jobs')
def autocomplete_jobs():
    query = request.args.get('q', '').strip().lower()
    if not query:
        return jsonify([])
    
    all_jobs = data['Job Title'].dropna().unique().tolist()
    matches = [j for j in all_jobs if query in j.lower()]
    return jsonify(sorted(matches)[:10])

@app.route("/add-to-todo", methods=["POST"])
@login_required
def add_to_todo():
    data_json = request.get_json()
    skill = html.unescape(data_json.get("skill", "").strip())
    job = html.unescape(data_json.get("job", "").strip())

    if not skill:
        return jsonify({"status": "error", "message": "No skill received"}), 400

    completed = current_user.get_completed_skills()
    if any(item['skill'].strip().lower() == skill.lower() for item in completed):
        return jsonify({"status": "success", "todo": current_user.get_todo_skills(), "completed": completed})

    todo = current_user.get_todo_skills()
    existing = next((item for item in todo if item['skill'].strip().lower() == skill.lower()), None)
    if existing:
        if job and job not in existing['jobs']:
            existing['jobs'].append(job)
    else:
        todo.append({
            "skill": skill,
            "jobs": [job] if job else []
        })

    current_user.set_todo_skills(todo)
    db.session.commit()

    return jsonify({
        "status": "success",
        "todo": todo,
        "completed": current_user.get_completed_skills()
    })

@app.route("/add-target-job", methods=["POST"])
@login_required
def add_target_job():
    data_json = request.get_json()
    job_title = html.unescape(data_json.get("job_title", "").strip())

    if not job_title:
        return jsonify({"status": "error", "message": "No job received"}), 400

    target_jobs = current_user.get_target_jobs()
    todo = current_user.get_todo_skills()
    completed = current_user.get_completed_skills()

    if job_title not in target_jobs:
        target_jobs.append(job_title)

    job_row = data[data['Job Title'].str.lower() == job_title.lower()]
    if not job_row.empty:
        skills = [s.strip() for s in job_row.iloc[0]['Skills'].split(',') if s.strip()]
        for skill in skills:
            if any(item['skill'].strip().lower() == skill.lower() for item in completed):
                continue
            existing = next((item for item in todo if item['skill'].strip().lower() == skill.lower()), None)
            if existing:
                if job_title not in existing['jobs']:
                    existing['jobs'].append(job_title)
            else:
                todo.append({
                    "skill": skill,
                    "jobs": [job_title]
                })

    current_user.set_target_jobs(target_jobs)
    current_user.set_todo_skills(todo)
    current_user.set_completed_skills(completed)
    db.session.commit()

    return jsonify({
        "status": "success",
        "target_jobs": target_jobs,
        "todo": todo,
        "completed": completed
    })

# ---------------- DATABASE SETUP ---------------- #

def create_tables():
    """Create database tables"""
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")

# ---------------- RUN APP ---------------- #
if __name__ == '__main__':
    create_tables()  # Create tables on startup
    app.run(debug=True)