# AI 어르신 돌봄 안전 모니터 (AI Home Care)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)
![Gemini](https://img.shields.io/badge/Google_Gemini-API-blueviolet?logo=google)
![Database](https://img.shields.io/badge/Database-SQLite-blue?logo=sqlite)
![Frontend](https://img.shields.io/badge/Frontend-Bootstrap_5-purple?logo=bootstrap)

**Google Gemini API**의 멀티모달(Multimodal) 분석 능력을 활용하여, 영상 속 어르신의 행동을 분석하고 '넘어짐'과 같은 잠재적 위험 상황을 감지하는 Python Flask 기반 웹 애플리케이션입니다.

사용자 인증(로그인) 기능을 통해 개인화된 분석 이력을 제공하며, 모든 분석 내용은 데이터베이스에 자동으로 기록되어 사용자가 언제든지 과거 상황을 다시 확인할 수 있도록 지원합니다.

## 📺 데모 스크린샷

* 메인 대시보드 (분석 결과 및 이력)
    ![메인 화면](image_f3a37c.jpg)
* 로그인 화면
    ![로그인 화면](image_bdf7b6.png)

## ✨ 주요 기능

* **🤖 AI 영상 분석 (Gemini Pro Vision):**
    * 업로드된 동영상 파일(.mp4, .webm)을 1초 간격의 프레임으로 자동 분할합니다.
    * Google Gemini API에 프레임들을 전송하여 '넘어짐', '위험' 등 어르신의 행동 패턴을 분석합니다.
* **🔐 사용자 인증 시스템 (Flask-Login):**
    * 회원가입 및 로그인 기능을 제공합니다.
    * 비밀번호는 **Bcrypt**로 안전하게 해시(Hash)되어 DB에 저장됩니다.
* **📊 개인화된 대시보드:**
    * 로그인한 사용자는 **본인이 업로드한 영상의 분석 이력**만 조회할 수 있습니다.
    * 분석 결과는 '상태', '상세 설명', '긴급 조치'로 구분되어 직관적으로 표시됩니다.
* **🗂️ 자동 분석 이력 관리 (History Log):**
    * 모든 분석 결과(썸네일, 상태, 설명, 타임스탬프)는 **SQLite** 데이터베이스에 자동으로 저장됩니다.
    * 이력 목록에서 특정 항목을 클릭하면 과거의 상세 분석 내용을 다시 확인할 수 있습니다.
* **🎨 반응형 UI (Bootstrap 5):**
    * 부트스트랩(Bootstrap)과 커스텀 CSS, 아이콘을 활용하여 깔끔하고 현대적인 UI를 구현했습니다.

## 🛠️ 기술 스택

| 분야 | 기술 | 설명 |
| :--- | :--- | :--- |
| **Backend** | `Flask` | 경량 파이썬 웹 프레임워크 |
| | `Flask-SQLAlchemy` | 데이터베이스(SQLite) ORM |
| | `Flask-Login` | 사용자 세션 및 인증 관리 |
| | `Flask-Bcrypt` | 비밀번호 해시(암호화) |
| **AI (Google)** | `google-generativeai` | Gemini Pro Vision API 연동 |
| **Video** | `opencv-python` | 비디오 파일 프레임 추출 및 썸네일 생성 |
| **Database** | `SQLite` | 로컬 데이터베이스 (분석 이력, 사용자 정보 저장) |
| **Frontend** | `HTML` / `Jinja2` | 웹 페이지 구조 |
| | `CSS` / `Bootstrap 5`| 그라데이션 배경, 카드, 아이콘 등 UI 디자인 |
| | `JavaScript (Fetch)`| 비동기 API 요청 (영상 업로드, 이력 로딩) |
| **Environment**| `python-dotenv` | API 키 등 환경 변수 관리 |

## 📁 프로젝트 구조

```
AI_Home_Care/
├── instance/
│   └── analysis_history.db     # (자동 생성) SQLite 데이터베이스 파일
├── static/                      # (향후 확장용) CSS/JS 폴더
├── temp_frames/                 # (자동 생성) AI 분석을 위한 임시 프레임
├── templates/                   # Flask가 렌더링할 HTML 파일
│   ├── index.html              # 메인 대시보드
│   ├── login.html              # 로그인 페이지
│   └── register.html           # 회원가입 페이지
├── thumbnails/                  # (자동 생성) 분석 이력 썸네일
├── uploads/                     # (자동 생성) 원본 업로드 영상
├── venv/                        # Python 가상 환경
├── .env                         # (직접 생성) API 키 저장
├── .gitignore                   # Git 제외 파일 목록
├── app.py                       # 메인 Flask 애플리케이션
├── requirements.txt             # 프로젝트 필요 라이브러리
└── README.md                    # 프로젝트 소개
```

## 🚀 시작하기

이 프로젝트를 로컬 환경에서 실행하는 방법입니다.

### 1. 리포지토리 복제

```bash
git clone https://github.com/mugi0810/AI_Home_Care.git
cd AI_Home_Care
```

### 2. 가상 환경 생성 및 활성화

```bash
# Python 가상 환경 생성
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. 필요 라이브러리 설치

`requirements.txt` 파일을 사용하여 모든 종속성을 한 번에 설치합니다.

```bash
pip install -r requirements.txt
```

### 4. .env 파일 설정

Google Gemini API 키가 필요합니다. 프로젝트 루트 폴더(`app.py`와 같은 위치)에 `.env` 파일을 새로 생성하고 아래 내용을 복사하여 붙여넣으세요.

```
GOOGLE_API_KEY="여러분의_GEMINI_API_KEY를_여기에_입력하세요"
```

### 5. Flask 서버 실행

```bash
python app.py
```

서버가 실행되면, 웹 브라우저를 열고 `http://127.0.0.1:5000` 주소로 접속하세요.

## 📝 사용 방법

1. **회원가입**: 최초 방문 시 회원가입 페이지에서 계정을 생성합니다.
2. **로그인**: 생성한 계정으로 로그인합니다.
3. **영상 업로드**: 메인 대시보드에서 어르신의 행동이 담긴 동영상 파일을 업로드합니다.
4. **분석 결과 확인**: AI가 자동으로 영상을 분석하여 결과를 표시합니다.
5. **이력 관리**: 과거 분석 이력은 자동으로 저장되며, 언제든지 다시 확인할 수 있습니다.

## 🔒 보안

- 모든 비밀번호는 Bcrypt를 사용하여 안전하게 해시 처리됩니다.
- 사용자는 본인이 업로드한 영상의 분석 결과만 조회할 수 있습니다.
- API 키는 `.env` 파일을 통해 안전하게 관리됩니다.

## 📧 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해주세요.

