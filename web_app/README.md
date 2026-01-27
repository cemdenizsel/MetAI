# Emotion Analysis App

A Next.js web application for AI-powered emotion analysis from video content.

## Features

- **Video Upload**: Drag-and-drop video upload with support for MP4, AVI, MOV, WebM, MKV, FLV
- **Emotion Detection**: AI-powered analysis detecting 7 emotion categories (happy, sad, angry, fear, surprise, disgust, neutral)
- **Mental Health Insights**: Wellness scoring and personalized recommendations
- **Visual Analytics**: Interactive charts showing emotion distribution over time
- **Job Tracking**: Background processing with real-time progress tracking
- **History Management**: View, filter, and manage past analyses
- **Dark Mode**: Full theme support with system preference detection

## Tech Stack

- **Framework**: Next.js 14 with App Router
- **Styling**: Tailwind CSS
- **UI Components**: Shadcn UI (Radix UI primitives)
- **State Management**: Zustand
- **Charts**: Recharts
- **Authentication**: JWT with httpOnly cookies

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Backend API running at `http://localhost:8084`

### Installation

```bash
# Install dependencies
npm install

# Copy environment variables
cp .env.example .env.local

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the app.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_URL` | Backend API URL | `http://localhost:8084` |

## Project Structure

```
web_app/
├── app/                    # Next.js App Router pages
│   ├── (auth)/            # Authentication pages (login, register)
│   ├── (app)/             # Protected app pages (dashboard, upload, history, settings)
│   ├── api/               # API route proxies
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Landing page
├── components/
│   ├── ui/                # Shadcn UI components
│   ├── layout/            # Layout components (navigation)
│   └── emotion/           # Emotion analysis components
├── lib/
│   ├── api/               # API client functions
│   ├── stores/            # Zustand stores
│   └── hooks/             # Custom React hooks
├── types/                 # TypeScript type definitions
├── middleware.ts          # Route protection middleware
└── next.config.js         # Next.js configuration
```

## API Integration

The app connects to a FastAPI backend with the following endpoints:

- `POST /auth/login` - User authentication
- `POST /auth/register` - User registration
- `GET /auth/profile` - Get user profile
- `POST /api/v1/emotion/analyze` - Direct video analysis
- `POST /api/v1/jobs/submit` - Submit async analysis job
- `GET /api/v1/jobs/{id}/status` - Get job status
- `GET /api/v1/jobs/{id}/result` - Get job result
- `GET /api/v1/jobs/my-jobs` - List user's jobs

## Development

```bash
# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run linting
npm run lint
```

## License

MIT
