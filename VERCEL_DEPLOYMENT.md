# Vercel Deployment Guide

This guide explains how to deploy the Book-Embedded RAG Chatbot to Vercel.

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install with `npm i -g vercel`
3. **Git Repository**: The project should be in a Git repository
4. **Environment Variables**: Required API keys and configuration

## Deployment Steps

### 1. Prepare Environment Variables

Before deploying, you need to set up the following environment variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_HOST=your_qdrant_host_url_here
DATABASE_URL=your_database_connection_string_here
NEON_DB_URL=your_neon_postgres_connection_string_here
SECRET_KEY=your_secret_key_here
```

### 2. Link Your Project

```bash
cd hackthonq4
vercel
```

Follow the prompts to link your project to a Vercel account and project.

### 3. Configure Build Settings

The `vercel.json` file is already configured for this monorepo:

- Frontend (Docusaurus): Serves static files from the `frontend` directory
- Backend (FastAPI): Serves API routes under `/api/*` from the `backend` directory

### 4. Deploy

```bash
vercel --prod
```

Or for a preview deployment:
```bash
vercel
```

## Architecture

The deployment follows a monorepo pattern:

```
┌─────────────────┐    ┌──────────────────┐
│   Frontend      │    │    Backend       │
│  (Docusaurus)   │    │   (FastAPI)      │
│                 │    │                  │
│  / (static)     │◄───┤  /api/* (API)    │
│                 │    │                  │
└─────────────────┘    └──────────────────┘
         │                       │
         └───────────────────────┘
              Vercel Platform
```

## Routes

- `/*` → Serves the Docusaurus frontend
- `/api/*` → Routes to the FastAPI backend

## Environment Variables Setup

In your Vercel dashboard:

1. Go to your project settings
2. Navigate to "Environment Variables"
3. Add the required variables from `.env.example`

## Troubleshooting

### Common Issues

1. **Build Failures**: Ensure all dependencies are in `requirements.txt` (backend) and `package.json` (frontend)

2. **API Route Issues**: Verify that `/api/*` routes are correctly configured in `vercel.json`

3. **Environment Variables**: Make sure all required environment variables are set in the Vercel dashboard

4. **Lambda Size Limits**: The configuration sets `maxLambdaSize` to 15mb to accommodate the Python dependencies

### Performance Tips

1. **Optimize Dependencies**: Only include necessary packages in `requirements.txt`
2. **Caching**: Leverage Vercel's built-in caching for static assets
3. **CDN**: Vercel automatically serves static assets through their global CDN

## Post-Deployment

After deployment:

1. Verify the frontend is accessible at your Vercel URL
2. Test API endpoints like `https://your-project.vercel.app/api/v1/health`
3. Check environment variables are properly configured
4. Test the chat functionality end-to-end

## Rollback

To rollback to a previous deployment:
1. Go to your project in the Vercel dashboard
2. Navigate to the "Deployments" tab
3. Click "Rollback" on the desired deployment

## Custom Domain

To connect a custom domain:
1. Go to your project settings in Vercel
2. Navigate to "Domains"
3. Add your custom domain and follow DNS configuration instructions