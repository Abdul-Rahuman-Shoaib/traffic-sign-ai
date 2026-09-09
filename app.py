"""
Vision-Based Traffic Sign Recognition and Road Safety Assistant
Main Flask Application Server
PSN Engineering College, Tirunelveli - Dept. of Computer Science & Engineering
Team: Abdul Rahuman Shoaib, Esakkiraja, Jebicson Francis, K.G. Maharajan (2026 - 2027)
"""
import os
import shutil
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, Response
import urllib.request
import urllib.parse
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from detection import detect_signs, process_video_file, SIGN_METADATA
from report import generate_pdf_report, generate_single_detection_pdf

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "static", "reports")
SAMPLE_FOLDER = os.path.join(BASE_DIR, "static", "samples")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "traffic-sign-ai-secret-key-2026")

# Database URI: defaults to reliable SQLite, with MySQL fallback if configured
db_url = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'database.db')}")
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please sign in to access the Road Safety Assistant."
login_manager.login_message_category = "info"


# ==========================================
# Database Models
# ==========================================
class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    detections = db.relationship("Detection", backref="user", cascade="all, delete-orphan", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Detection(db.Model):
    __tablename__ = "detection_history"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    label = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ==========================================
# Authentication Routes
# ==========================================
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Username and password are required.", "danger")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("Username already taken. Please choose another.", "warning")
            return redirect(url_for("register"))

        user = User(username=username)
        user.set_password(password)

        # Make first registered user admin automatically
        if User.query.count() == 0:
            user.is_admin = True

        db.session.add(user)
        db.session.commit()
        flash("Operator account registered successfully! Please sign in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=True)
            flash(f"Welcome back, {user.username}! Road Safety Assistant is active.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username or password credentials.", "danger")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been signed out safely.", "info")
    return redirect(url_for("login"))


# ==========================================
# Core Application Routes
# ==========================================
@app.route("/")
def index():
    return redirect(url_for("dashboard")) if current_user.is_authenticated else redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    recent = Detection.query.filter_by(user_id=current_user.id).order_by(Detection.timestamp.desc()).limit(6).all()
    total_count = Detection.query.filter_by(user_id=current_user.id).count()
    warning_count = Detection.query.filter(
        Detection.user_id == current_user.id,
        Detection.label.ilike("%Caution%") | Detection.label.ilike("%Warning%") | Detection.label.ilike("%Stop%") | Detection.label.ilike("%School%")
    ).count()

    return render_template(
        "dashboard.html",
        recent=recent,
        total_user_detections=total_count,
        warning_count=warning_count,
        sign_catalog=SIGN_METADATA
    )


@app.route("/detect", methods=["POST"])
@login_required
def detect():
    """Processes uploaded image or webcam blob and saves detection results."""
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No visual file provided."}), 400

    filename = secure_filename(f"{current_user.id}_{int(datetime.utcnow().timestamp())}_{file.filename or 'frame.jpg'}")
    if not (filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")):
        filename += ".jpg"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    results = detect_signs(filepath)

    saved_records = []
    for r in results:
        d = Detection(
            user_id=current_user.id,
            image_path=f"uploads/{filename}",
            label=r["label"],
            confidence=r["confidence"]
        )
        db.session.add(d)
        saved_records.append(r)

    db.session.commit()

    return jsonify({
        "status": "success",
        "results": saved_records,
        "image_url": url_for("static", filename=f"uploads/{filename}")
    })


@app.route("/detect-video", methods=["POST"])
@login_required
def detect_video():
    """Processes an uploaded video file and returns keyframe detection timeline."""
    video_file = request.files.get("video")
    if not video_file:
        return jsonify({"error": "No video file provided."}), 400

    filename = secure_filename(f"v_{current_user.id}_{int(datetime.utcnow().timestamp())}_{video_file.filename}")
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    video_file.save(video_path)

    result_data = process_video_file(video_path, app.config["UPLOAD_FOLDER"], sample_rate=2)

    # Save sampled detections to DB
    if "timeline" in result_data:
        for item in result_data["timeline"]:
            d = Detection(
                user_id=current_user.id,
                image_path=item["frame_image"],
                label=item["label"],
                confidence=item["confidence"]
            )
            db.session.add(d)
        db.session.commit()

    return jsonify(result_data)


@app.route("/live-webcam")
@login_required
def live_webcam():
    """Standalone live webcam driver assistant HUD page."""
    return render_template("dashboard.html", active_tab="webcam", sign_catalog=SIGN_METADATA)


@app.route("/history")
@login_required
def history():
    """Searchable audit log of detections."""
    detections = Detection.query.filter_by(user_id=current_user.id).order_by(Detection.timestamp.desc()).all()
    return render_template("history.html", detections=detections)


@app.route("/history/delete/<int:detection_id>")
@login_required
def delete_detection(detection_id):
    detection = Detection.query.get_or_404(detection_id)
    if detection.user_id == current_user.id or current_user.is_admin:
        db.session.delete(detection)
        db.session.commit()
        flash("Detection record deleted.", "info")
    return redirect(url_for("history"))


@app.route("/history/clear")
@login_required
def clear_history():
    Detection.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash("Your detection history has been cleared.", "success")
    return redirect(url_for("history"))


@app.route("/catalog")
@login_required
def catalog():
    """Educational road sign specifications and safety rules."""
    return render_template("catalog.html", sign_catalog=SIGN_METADATA)


@app.route("/about")
def about():
    """Academic project presentation and team showcase."""
    return render_template("about.html")


@app.route("/report/pdf")
@login_required
def report_pdf():
    """Generates official academic PDF report."""
    detections = Detection.query.filter_by(user_id=current_user.id).order_by(Detection.timestamp.desc()).all()
    pdf_path = generate_pdf_report(current_user.username, detections)
    return send_file(pdf_path, as_attachment=True, download_name=f"Traffic_Sign_Report_{current_user.username}.pdf")


@app.route("/report/pdf/<int:detection_id>")
@login_required
def report_single_pdf(detection_id):
    """Generates individual detection inspection certificate PDF."""
    detection = Detection.query.get_or_404(detection_id)
    pdf_path = generate_single_detection_pdf(detection, current_user.username)
    return send_file(pdf_path, as_attachment=True, download_name=f"Inspection_Sheet_#{detection.id}.pdf")


# ==========================================
# Admin Routes
# ==========================================
def admin_required(func):
    from functools import wraps
    @wraps(func)
    @login_required
    def wrapper(*args, **kwargs):
        if not current_user.is_admin:
            flash("Administrator access required for this resource.", "danger")
            return redirect(url_for("dashboard"))
        return func(*args, **kwargs)
    return wrapper


@app.route("/admin")
@admin_required
def admin_panel():
    users = User.query.all()
    total_detections = Detection.query.count()
    return render_template("admin.html", users=users, total_detections=total_detections)


@app.route("/admin/user/<int:user_id>/toggle-admin")
@admin_required
def admin_toggle_role(user_id):
    user = User.query.get_or_404(user_id)
    if user.id != current_user.id:
        user.is_admin = not user.is_admin
        db.session.commit()
        flash(f"Updated role for {user.username}.", "success")
    return redirect(url_for("admin_panel"))


@app.route("/admin/user/<int:user_id>/delete")
@admin_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id != current_user.id:
        db.session.delete(user)
        db.session.commit()
        flash(f"Deleted user account {user.username}.", "success")
    return redirect(url_for("admin_panel"))


@app.route("/api/tts")
def api_tts():
    text = request.args.get("text", "").strip()
    if not text:
        return ("", 400)
    try:
        encoded_text = urllib.parse.quote(text[:250])
        tts_url = f"https://translate.google.com/translate_tts?ie=UTF-8&tl=en&client=tw-ob&q={encoded_text}"
        req = urllib.request.Request(tts_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            audio_data = resp.read()
        return Response(audio_data, mimetype="audio/mpeg")
    except Exception as e:
        return (f"TTS error: {e}", 500)


# ==========================================
# Database Seeding & Initialization Helper
# ==========================================
def seed_database():
    with app.app_context():
        db.create_all()

        # Seed Admin User if not present
        if not User.query.filter_by(username="admin").first():
            admin = User(username="admin", is_admin=True)
            admin.set_password("admin123")
            db.session.add(admin)

        # Seed Student User if not present
        if not User.query.filter_by(username="student").first():
            student = User(username="student", is_admin=False)
            student.set_password("student123")
            db.session.add(student)

        db.session.commit()

        # Seed sample detection history for admin and student if empty
        if Detection.query.count() == 0:
            admin_user = User.query.filter_by(username="admin").first()
            if admin_user:
                # Copy sample images to uploads
                for sample_file, label, conf in [
                    ("sample_stop.jpg", "Stop Sign", 0.96),
                    ("sample_speed_50.jpg", "Speed Limit (50 km/h)", 0.93),
                    ("sample_pedestrian.jpg", "Pedestrian Crossing", 0.90),
                    ("sample_yield.jpg", "Yield / Give Way", 0.91),
                    ("sample_school_zone.jpg", "School Zone / Children Crossing", 0.92)
                ]:
                    src = os.path.join(SAMPLE_FOLDER, sample_file)
                    dst = os.path.join(UPLOAD_FOLDER, sample_file)
                    if os.path.exists(src):
                        shutil.copyfile(src, dst)
                    d = Detection(
                        user_id=admin_user.id,
                        image_path=f"uploads/{sample_file}",
                        label=label,
                        confidence=conf
                    )
                    db.session.add(d)
                db.session.commit()

# Ensure database tables and initial seed data are populated on startup
with app.app_context():
    seed_database()

def get_local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

if __name__ == "__main__":
    seed_database()
    local_ip = get_local_ip()
    print("======================================================================")
    print("Road Safety Assistant Server Running!")
    print("  Local PC URL:   http://127.0.0.1:5000")
    print(f"  Mobile/Wi-Fi:   http://{local_ip}:5000")
    print("======================================================================")
    app.run(host="0.0.0.0", port=5000, debug=True)

