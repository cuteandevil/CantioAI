# CantioAI Web Interface

This is the frontend for the CantioAI Web Interface, built with React 18, TypeScript, Ant Design, and Tailwind CSS.

## Development

To start the development server:

```bash
npm install
npm run dev
```

The application will be available at http://localhost:3000

## Build

To build the application for production:

```bash
npm run build
```

The built files will be in the `dist` directory.

## Features

- Model management
- Audio file upload and processing
- Real-time audio processing
- Result management and history
- Responsive design
- Dark/Light theme support

## Technology Stack

- React 18
- TypeScript
- Vite
- Ant Design 5.x
- Tailwind CSS
- WaveSurfer.js (for audio visualization)
- Axios (for API communication)

## API Endpoints

The frontend communicates with the backend API running on port 7860:

- GET /api/health - Health check
- GET /api/models - List available models
- POST /api/models/{id}/load - Load a model
- POST /api/audio/upload - Upload audio file
- POST /api/audio/process - Process audio
- GET /api/audio/status/{task_id} - Get task status
- WebSocket: ws://localhost:7860/ws/realtime/{client_id} - Real-time processing