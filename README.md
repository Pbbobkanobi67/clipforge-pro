# ClipForge Pro

AI-powered video analysis platform for creating viral clips with automated transcription, scene detection, and smart export features.

## Features

### Core Analysis
- 🎬 **AI Video Analysis** - Automatic transcription, speaker diarization, and scene detection
- ⚡ **Viral Clip Detection** - AI scores clips for virality potential
- 🎯 **Hook Detection** - Identifies compelling opening sequences
- 👁️ **Visual Analysis** - YOLO-powered object and person detection

### Pro Features (OpusClip-style)
- ✨ **Animated Captions** - Karaoke, bounce, typewriter, fade effects
- 📐 **AI Reframe** - Smart cropping for 9:16, 1:1, 16:9 formats
- 🎨 **Brand Templates** - Logo overlays, colors, and outros
- 🖼️ **Thumbnail Generator** - AI-scored frames with text overlays
- 🎞️ **Timeline Editor** - Multi-clip editing and rendering
- 📤 **Export to XML** - FCPXML, Premiere Pro, and EDL formats

### Utilities
- 📁 **Preset Management** - Save and manage download configurations
- 🕐 **Timestamp Converter** - Convert between formats
- 📜 **Command History** - Track generated commands

## Architecture

```
ClipForge_Pro/
├── index.html              # React frontend (static)
├── backend/
│   ├── app/
│   │   ├── api/endpoints/  # FastAPI routes
│   │   ├── models/         # SQLAlchemy models, Pydantic schemas
│   │   ├── services/       # AI/ML services
│   │   └── main.py         # Application entry
│   ├── Dockerfile          # Docker with GPU support
│   └── pyproject.toml      # Python dependencies
├── vercel.json             # Frontend deployment config
└── README.md
```

## Deployment

### Frontend (Vercel)
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel

# Production deploy
vercel --prod
```

### Backend (Railway/Render/VPS)

#### Option 1: Docker
```bash
cd backend
docker build -t clipforge-pro-backend .
docker run -p 8000:8000 --gpus all clipforge-pro-backend
```

#### Option 2: Direct Python
```bash
cd backend
python -m venv .venv
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -e .
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Environment Variables
```env
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/clipforge

# LLM (optional, for enhanced analysis)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Storage
STORAGE_PATH=/app/storage

# CORS (set to your frontend URL)
CORS_ORIGINS=["https://your-app.vercel.app"]
```

## API Documentation

Once the backend is running, access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/videos/upload` | Upload video file |
| `POST /api/v1/videos/download` | Download from URL |
| `POST /api/v1/analysis/{video_id}/start` | Start AI analysis |
| `GET /api/v1/analysis/{job_id}/results` | Get analysis results |
| `GET /api/v1/clips/{job_id}` | Get clip suggestions |
| `POST /api/v1/clips/{clip_id}/export` | Export clip |
| `POST /api/v1/captions/styles` | Create caption style |
| `POST /api/v1/reframe/{clip_id}` | Generate AI reframe |
| `POST /api/v1/thumbnails/{clip_id}/generate` | Generate thumbnails |

## Tech Stack

### Frontend
- React 18 (CDN, no build required)
- Vanilla CSS with CSS variables

### Backend
- FastAPI (async Python)
- SQLAlchemy (async ORM)
- faster-whisper (transcription)
- pyannote.audio (speaker diarization)
- YOLO v8 (visual analysis)
- OpenCV (scene detection)
- FFmpeg (video processing)

## License

MIT
