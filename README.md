# Video Extract Pro

A powerful yt-dlp command generator with presets, timestamp converter, and command history.

## Features

- 🎬 **Video URL Input** - Supports YouTube, Vimeo, Twitter, TikTok, and more
- ⚡ **Quick Generate** - One-click command generation with default preset
- 📁 **Preset Management** - Save and manage download configurations
- 🕐 **Timestamp Converter** - Convert between YouTube Chapters, Timecode, and FFMPEG metadata
- 📜 **Command History** - Track your last 50 generated commands
- 🔧 **Advanced Options** - Subtitles, SponsorBlock, rate limiting, and more

## Deploy to Vercel (Easy Method)

### Option 1: Drag & Drop
1. Go to [vercel.com](https://vercel.com)
2. Sign in (or create account)
3. Click "Add New..." → "Project"
4. Drag and drop the `video-extract-pro` folder
5. Click "Deploy"
6. Done! Your app will be live at `your-project.vercel.app`

### Option 2: Vercel CLI
```bash
# Install Vercel CLI
npm install -g vercel

# Navigate to project folder
cd video-extract-pro

# Deploy
vercel

# Follow prompts, then deploy to production
vercel --prod
```

### Option 3: GitHub + Vercel
1. Create a new GitHub repository
2. Push this project to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/video-extract-pro.git
   git push -u origin main
   ```
3. Go to [vercel.com](https://vercel.com)
4. Click "Add New..." → "Project"
5. Import your GitHub repository
6. Click "Deploy"

## Other Hosting Options

### GitHub Pages
1. Push to GitHub
2. Go to repo Settings → Pages
3. Select "main" branch, root folder
4. Save - your site will be at `username.github.io/video-extract-pro`

### Netlify
1. Go to [netlify.com](https://netlify.com)
2. Drag and drop the folder
3. Done!

### Local Development
```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx serve .

# Using PHP
php -S localhost:8000
```

Then open `http://localhost:8000` in your browser.

## Project Structure

```
video-extract-pro/
├── index.html      # Main application (single-file React app)
├── package.json    # Project metadata
├── vercel.json     # Vercel deployment config
└── README.md       # This file
```

## Usage

1. Paste a YouTube (or other platform) URL
2. Select a preset or customize options
3. Click "Quick Generate" or copy the generated command
4. Run the command in your terminal (requires yt-dlp installed)

### Install yt-dlp
```bash
# macOS
brew install yt-dlp

# Windows
winget install yt-dlp

# Linux
sudo apt install yt-dlp
# or
pip install yt-dlp
```

## License

MIT
