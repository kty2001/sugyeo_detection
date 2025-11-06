
import os
import sys
import uvicorn
from pathlib import Path

if any("--multiprocessing-fork" in arg for arg in sys.argv):
    with open("trace.log", "a", encoding="utf-8") as f:
        f.write("Detected multiprocessing fork process. Skipping server startup.\n")
    sys.exit(0)

try:
    from app.main import app
except Exception as e:
    with open("import_error.log", "w", encoding="utf-8") as f:
        f.write(f"app import 실패: {e}\n")
    raise
    
with open("trace.log", "a", encoding="utf-8") as f:
    f.write(f"sys.argv: {sys.argv}\n")
    f.write(f"sys.executable: {sys.executable}\n")
    f.write(f"__file__: {__file__}\n")
    f.write(f"cwd: {os.getcwd()}\n")

if __name__ == "__main__":
    # 실행 파일 경로 기준으로 작업 디렉토리 설정
    if getattr(sys, 'frozen', False):
        # PyInstaller로 패키징된 경우
        base_dir = Path(sys._MEIPASS)
        os.chdir(base_dir)

    if getattr(sys, 'frozen', False):
        base_dir = Path(sys._MEIPASS)
        os.chdir(base_dir)
        with open("trace.log", "a", encoding="utf-8") as f:
            f.write(f"[AFTER CHDIR] cwd: {os.getcwd()}\n")    

    with open("startup.log", "w", encoding="utf-8") as f:
        f.write("FastAPI EXE started!\n")

    try:        
        # 결과 및 업로드 디렉토리 생성
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        # 로깅 설정 - 콘솔 출력 비활성화
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(levelprefix)s %(message)s",
                    "use_colors": False,
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.NullHandler",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "INFO"},
            },
        }
        
        # FastAPI 서버 시작 - 로깅 설정 추가
        print("FastAPI 서버를 시작합니다...")
        uvicorn.run(app, host="127.0.0.1", port=8000, log_config=log_config)
    except Exception as e:
        with open("error.log", "w", encoding="utf-8") as f:
            f.write(f"Error starting FastAPI server: {e}\n")
