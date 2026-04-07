# ClipForge Pro

AI-powered video analysis and clip extraction tool. Detects viral moments, generates captions, and exports production-ready clips.

## Quick Start

### Local Development
```bash
# 1. Start backend (requires Python 3.10+, FFmpeg, CUDA GPU recommended)
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000

# 2. Open frontend
# Either open index.html directly, or visit https://clipforge-pro.vercel.app
```

### Online Access (Cloudflare Tunnel)
```bash
# 1. Start backend locally
cd backend
python -m uvicorn app.main:app --port 8000

# 2. Start Cloudflare Tunnel (free, no account needed)
cloudflared tunnel --url http://localhost:8000
# Prints: https://<random>.trycloudflare.com

# 3. Visit https://clipforge-pro.vercel.app
# Paste the tunnel URL when prompted → click Connect
```

Install cloudflared: `winget install cloudflare.cloudflared` (Windows) or `brew install cloudflared` (macOS)

## Architecture

```
Frontend (Vercel)          Tunnel (Cloudflare)         Backend (Local PC)
clipforge-pro.vercel.app → *.trycloudflare.com    →   localhost:8000
     React SPA                 Free tunnel                FastAPI
     index.html                cloudflared               FFmpeg + Whisper
                                                          SQLite + CUDA
```

- **Frontend**: Single-file React SPA (`index.html`), deployed on Vercel
- **Backend**: FastAPI + SQLAlchemy + SQLite, runs locally with GPU access
- **Tunnel**: Cloudflare Tunnel (free) bridges the two — new URL each restart
- **CORS**: Backend allows `*.trycloudflare.com` via regex + explicit Vercel domain

## Key Directories

| Path | Purpose |
|------|---------|
| `backend/app/` | FastAPI application code |
| `backend/app/services/` | Core services (video, caption, analysis) |
| `backend/app/api/endpoints/` | REST API endpoints |
| `backend/storage/` | SQLite DB, uploaded videos, exported clips |
| `index.html` | Complete frontend (React + Babel, no build step) |

## Environment Variables

Set in `backend/.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API for AI analysis |
| `HF_TOKEN` | Yes | Hugging Face token for pyannote diarization |
| `PEXELS_API_KEY` | No | Pexels API for B-roll stock footage |
| `PIXABAY_API_KEY` | No | Pixabay API for B-roll stock footage |
| `WHISPER_MODEL` | No | Whisper model size (default: `small`) |
| `WHISPER_DEVICE` | No | `cuda` or `cpu` (default: `cuda`) |

## Future Upgrade Paths

### Option A: Free Cloud Hosting
- Deploy backend on Render.io free tier (no GPU)
- Replace local Whisper with OpenAI Whisper API (~$0.006/min)
- Keep SQLite, use Render's persistent disk
- Cost: ~$0.10-0.50 per video in API costs, $0 hosting

### Option B: Paid Cloud Hosting ($7-25/mo)
- Render or Railway with PostgreSQL + persistent storage
- More reliable than free tier, no cold starts
- Still use OpenAI Whisper API for transcription

### Option C: Full SaaS
- Auth: JWT + user accounts (Supabase or Auth0)
- Payments: Stripe subscriptions (Free/Pro/Business tiers)
- GPU: Modal or RunPod for on-demand processing
- Storage: S3 for videos + CloudFront CDN for delivery
- Database: PostgreSQL (Supabase or Neon)
