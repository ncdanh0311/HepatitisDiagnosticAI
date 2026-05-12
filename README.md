# Hepatitis AI — Healthcare Analytics Platform

Nền tảng phân tích và chẩn đoán viêm gan sử dụng AI, được chuyển đổi từ bản demo Streamlit sang hệ thống Fullstack chuyên nghiệp.

## 🚀 Tech Stack Cơ Bản

- **Frontend**: React (Vite) + TailwindCSS
- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL + Redis
- **AI/ML**: Scikit-learn + LangChain (RAG Chatbot)
- **DevOps**: Docker & Docker Compose

## 🛠️ Cài đặt nhanh

### 1. Cấu hình môi trường
```bash
cp .env.example .env
# Chỉnh sửa .env và thêm GEMINI_API_KEY nếu dùng Chatbot
```

### 2. Chạy ứng dụng (Docker)
```bash
docker-compose up -d
```

### 3. Khởi tạo Database
```bash
cd backend
# (Tạo venv và cài đặt dependencies nếu chạy local)
alembic upgrade head
```

## 📂 Cấu trúc dự án
- `frontend/`: Giao diện người dùng React.
- `backend/`: API chính xử lý logic và người dùng.
- `ml-service/`: Dịch vụ dự đoán AI.
- `rag-service/`: Chatbot tư vấn y tế.
- `shared/`: Chứa Model AI và dữ liệu dùng chung.
- `database/`: Scripts khởi tạo cơ sở dữ liệu.

---
*Truy cập Dashboard tại: http://localhost:5173*
