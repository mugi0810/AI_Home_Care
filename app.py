# app.py

# --- 1. 모든 라이브러리 임포트 ---
import os
import cv2
import json
import shutil
import uuid
import datetime

from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory, flash, redirect, url_for # ⭐️ flash, redirect, url_for 추가
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user # ⭐️ 추가
from flask_bcrypt import Bcrypt # ⭐️ 추가

# --- 2. 헬퍼 함수 정의 ---
# (이 함수들이 API 엔드포인트보다 *반드시* 먼저 정의되어야 합니다)

def extract_frames(video_path, output_folder, frame_interval=30):
    """
    비디오에서 30프레임(약 1초) 간격으로 이미지를 추출합니다.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vid_cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frame_count = 0
    frame_files = []

    while vid_cap.isOpened():
        success, image = vid_cap.read()
        if not success:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, image)
            frame_files.append(frame_filename)
            saved_frame_count += 1
            
        frame_count += 1
        
    vid_cap.release()
    return frame_files

def extract_thumbnail(video_path, thumbnail_path):
    """
    비디오의 첫 번째 프레임을 썸네일로 저장합니다.
    """
    vid_cap = cv2.VideoCapture(video_path)
    success, image = vid_cap.read()
    if success:
        cv2.imwrite(thumbnail_path, image)
    vid_cap.release()
    return thumbnail_path

# --- 3. 앱 설정 및 초기화 ---

# .env 파일에서 환경 변수(API 키) 로드
load_dotenv()

app = Flask(__name__)

# ⭐️ (필수) flash 메시지를 사용하려면 secret_key가 필요합니다.
app.config['SECRET_KEY'] = 'dev_secret_key_please_change_this' 

# DB 및 업로드 폴더 설정
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///analysis_history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['THUMBNAIL_FOLDER'] = 'thumbnails'

# Gemini API 키 설정
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("오류: GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다.")
    exit()

# DB 객체 초기화
db = SQLAlchemy(app)
bcrypt = Bcrypt(app) # ⭐️ Bcrypt 초기화

# ⭐️ Flask-Login 설정 시작 ⭐️
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # ⭐️ 로그인이 필요한 페이지 접근 시 'login' 라우트로 리디렉션
login_manager.login_message = "로그인이 필요한 서비스입니다."
login_manager.login_message_category = "info" # (부트스트랩 alert 스타일용)

@login_manager.user_loader
def load_user(user_id):
    """세션에서 사용자 ID를 받아 User 객체를 반환"""
    return User.query.get(int(user_id))
# ⭐️ Flask-Login 설정 끝 ⭐️


# --- 4. 데이터베이스 모델 정의 ---

# ⭐️ 새로운 User 모델 (UserMixin 상속 필수) ⭐️
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # ⭐️ 'AnalysisLog' 모델의 'author' 필드와 연결됩니다.
    analysis_logs = db.relationship('AnalysisLog', backref='author', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        """비밀번호를 해시하여 저장"""
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        """해시된 비밀번호와 입력된 비밀번호를 비교"""
        return bcrypt.check_password_hash(self.password_hash, password)

class AnalysisLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    video_filename = db.Column(db.String(255), nullable=False)
    thumbnail_filename = db.Column(db.String(255), nullable=False)
    
    # Gemini 분석 결과
    status = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    is_urgent = db.Column(db.Boolean, default=False, nullable=False)
    
    # ⭐️⭐️⭐️ (수정됨) ⭐️⭐️⭐️
    # 'user.id' (user 테이블의 id 컬럼)와 연결하는 외래 키
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    # 결과를 dict 형태로 변환 (JSON 응답용)
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() + 'Z',
            'thumbnail_url': f'/thumbnails/{self.thumbnail_filename}',
            'status': self.status,
            'description': self.description,
            'is_urgent': self.is_urgent
        }

# --- 5. Gemini 분석 함수 ---
def analyze_video_frames_with_gemini(frame_files):
    """
    추출된 프레임 목록을 Gemini API로 전송하고 분석 결과를 받습니다.
    """
    
    # 1. 모델 설정 (⭐️ 버그 수정: 'gemini-2.0-flash' -> 'gemini-2.0-flash')
    model = genai.GenerativeModel('gemini-2.5-flash')

    # 2. 프롬프트 엔지니어링 (JSON 응답 유도)
    prompt = """
    당신은 AI 어르신 안전 모니터링 전문가입니다.
    아래에 시간 순서대로 제공되는 이미지 프레임들을 분석해주세요.
    
    작업:
    1. 어르신의 행동을 분석하여 '넘어짐' 또는 '위급 상황'이 감지되는지 확인합니다.
    2. 분석 결과를 바탕으로 '상태', '상세 설명', '긴급 조치 필요 여부'를 판단합니다.
    3. 응답은 반드시 아래의 JSON 형식으로만 제공해야 합니다.

    JSON 형식:
    {
      "status": "정상" or "주의" or "위험",
      "description": "어르신이 소파에 앉아 TV를 시청하고 있습니다." or "어르신이 바닥에 쓰러져 움직이지 않습니다.",
      "is_urgent": false or true
    }

    이제 이미지 프레임 분석을 시작합니다.
    """

    # 3. 이미지 파일과 프롬프트 준비
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json" # JSON 응답 강제
    )

    prompt_parts = [prompt]
    opened_images = [] # 열린 파일 핸들 추적

    try:
        for frame_file in frame_files:
            img = Image.open(frame_file)
            prompt_parts.append(img)
            opened_images.append(img) # 리스트에 추가

        # 4. API 호출
        response = model.generate_content(
            prompt_parts,
            generation_config=generation_config,
            request_options={'timeout': 600} # 타임아웃 10분
        )

        # 5. JSON 응답 파싱
        result = json.loads(response.text)
        return result

    except Exception as e:
        print(f"Gemini API 호출 오류: {e}")
        return None

    finally:
        # 1. 열려있는 모든 이미지 파일 핸들을 먼저 닫습니다. (PermissionError 방지)
        for img in opened_images:
            img.close()

        # 2. 이제 파일을 닫았으므로 임시 폴더를 삭제합니다.
        if frame_files: # 프레임이 하나라도 생성되었을 경우에만 실행
            temp_frame_folder = os.path.dirname(frame_files[0])
            if os.path.exists(temp_frame_folder):
                shutil.rmtree(temp_frame_folder)
                
# --- 6. Flask API 엔드포인트 (웹사이트 주소) ---

@app.route('/')
@login_required  # ⭐️ 1. 메인 페이지에 로그인 요구
def index():
    """메인 HTML 페이지를 렌더링합니다."""
    # ⭐️ 2. index.html에 사용자 이름을 전달
    return render_template('index.html', username=current_user.username)

@app.route('/analyze', methods=['POST'])
@login_required # ⭐️ 3. 분석 API에도 로그인 요구
def analyze_video():
    """
    영상 파일을 받아 분석하고, 결과를 DB에 저장한 후 JSON으로 반환합니다.
    """
    if 'video' not in request.files:
        return jsonify({'error': '비디오 파일이 없습니다.'}), 400

    # ... (파일 업로드 로직 동일) ...
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다.'}), 400

    unique_id = str(uuid.uuid4())
    video_ext = os.path.splitext(video_file.filename)[1].lower() 
    video_filename = f"{unique_id}{video_ext}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    
    thumbnail_filename = f"{unique_id}.jpg"
    thumbnail_path = os.path.join(app.config['THUMBNAIL_FOLDER'], thumbnail_filename)
    temp_frame_folder = os.path.join('temp_frames', unique_id) 

    try:
        video_file.save(video_path)
        extract_thumbnail(video_path, thumbnail_path)
        frame_files = extract_frames(video_path, temp_frame_folder, frame_interval=30)
        
        if not frame_files:
            return jsonify({'error': '비디오에서 프레임을 추출할 수 없습니다.'}), 500

        analysis_result = analyze_video_frames_with_gemini(frame_files)
        
        if not analysis_result:
            return jsonify({'error': 'Gemini 분석에 실패했습니다.'}), 500

        # ⭐️ 4. (중요) new_log 생성 시 'user_id' 추가
        new_log = AnalysisLog(
            video_filename=video_filename,
            thumbnail_filename=thumbnail_filename,
            status=analysis_result.get('status', '오류'),
            description=analysis_result.get('description', '분석 실패'),
            is_urgent=analysis_result.get('is_urgent', False),
            user_id=current_user.id  # ⭐️⭐️⭐️ 현재 로그인한 사용자의 ID를 저장
        )
        db.session.add(new_log)
        db.session.commit()

        return jsonify(new_log.to_dict()), 201

    except Exception as e:
        if os.path.exists(temp_frame_folder):
            shutil.rmtree(temp_frame_folder)
        print(f"분석 중 심각한 오류 발생: {e}")
        return jsonify({'error': f'서버 내부 오류: {e}'}), 500


@app.route('/history', methods=['GET'])
@login_required # ⭐️ 5. 히스토리 API에도 로그인 요구
def get_history():
    """
    (수정) *현재 로그인한 사용자*의 분석 이력을 최신순으로 반환합니다.
    """
    # ⭐️ 6. (중요) DB 쿼리 수정!
    logs = AnalysisLog.query.filter_by(user_id=current_user.id).order_by(AnalysisLog.timestamp.desc()).all()
    return jsonify([log.to_dict() for log in logs])

@app.route('/history/<int:log_id>', methods=['GET'])
@login_required # ⭐️ 7. 상세 보기 API에도 로그인 요구
def get_history_detail(log_id):
    """
    (수정) 특정 이력을 반환하되, *반드시* 본인 소유인지 확인합니다.
    """
    log = AnalysisLog.query.get_or_404(log_id)
    
    # ⭐️ 8. 본인 것 아니면 403 에러
    if log.user_id != current_user.id:
        return jsonify({'error': '권한이 없습니다.'}), 403 
        
    return jsonify(log.to_dict())

@app.route('/thumbnails/<filename>')
def get_thumbnail(filename):
    """썸네일 이미지를 서빙합니다."""
    return send_from_directory(app.config['THUMBNAIL_FOLDER'], filename)


# ⭐️ --- 6-1. 인증 라우트 (새로 추가) --- ⭐️

@app.route('/register', methods=['GET', 'POST'])
def register():
    """회원가입 라우트"""
    if current_user.is_authenticated:
        return redirect(url_for('index')) # 이미 로그인했다면 메인으로

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("이미 존재하는 사용자명입니다.", "danger")
            return redirect(url_for('register'))

        new_user = User(username=username)
        new_user.set_password(password) # 비밀번호 해시
        
        db.session.add(new_user)
        db.session.commit()
        
        flash("회원가입이 완료되었습니다. 로그인해주세요.", "success")
        return redirect(url_for('login')) # 회원가입 성공 시 로그인 페이지로

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """로그인 라우트"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user, remember=True) # ⭐️ Flask-Login을 통해 로그인 처리
            return redirect(url_for('index'))
        else:
            flash("사용자명 또는 비밀번호가 올바르지 않습니다.", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
@login_required # 로그아웃은 로그인 한 사용자만 가능
def logout():
    """로그아웃 라우트"""
    logout_user() # ⭐️ Flask-Login을 통해 로그아웃 처리
    return redirect(url_for('login'))

# --- 7. 앱 실행 ---
if __name__ == '__main__':
    # (앱 실행 전 DB 및 폴더 생성)
    with app.app_context():
        db.create_all() # ⭐️ User 테이블과 수정된 AnalysisLog 테이블 생성
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['THUMBNAIL_FOLDER'], exist_ok=True)
        os.makedirs('temp_frames', exist_ok=True) # 임시 프레임 폴더
        
    app.run(debug=True, host='0.0.0.0', port=5000)