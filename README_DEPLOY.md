# Deploying Paper Summarizer to Render

This guide explains how to deploy the Paper Summarizer application to Render.

## Prerequisites

1. A [Render](https://render.com) account
2. A GitHub repository with the code
3. A Google Gemini API key

## Deployment Steps

### 1. Push Code to GitHub

First, create a new GitHub repository and push your code:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/paper-summarizer.git
git push -u origin main
```

### 2. Create New Web Service on Render

1. Log in to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: paper-summarizer (or your preferred name)
   - **Runtime**: Python
   - **Branch**: main
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT app_web:app`

### 3. Set Environment Variables

In the Render dashboard, add these environment variables:

- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `SECRET_KEY`: A random secret key for Flask sessions (optional, but recommended)
- `FLASK_ENV`: Set to `production`

To generate a secure secret key:
```python
import secrets
print(secrets.token_hex(32))
```

### 4. Deploy

1. Click "Create Web Service"
2. Render will automatically build and deploy your application
3. Wait for the deployment to complete (usually 2-5 minutes)
4. Your app will be available at `https://your-app-name.onrender.com`

## Alternative: Using render.yaml

If you prefer Infrastructure as Code, the project includes a `render.yaml` file. To use it:

1. Push the code to GitHub
2. Go to Render Dashboard
3. Click "New +" → "Blueprint"
4. Connect your GitHub repository
5. Render will automatically detect the `render.yaml` file
6. Add the GEMINI_API_KEY in the environment variables section
7. Deploy

## Important Notes

### Environment Variables
- **Never commit your API keys** to the repository
- Always use environment variables for sensitive data
- The app will show a warning if GEMINI_API_KEY is not set

### File Uploads
- Render's free tier has ephemeral storage
- Uploaded files will be deleted when the service restarts
- For production use, consider using external storage (S3, etc.)

### Performance
- The free tier may sleep after 15 minutes of inactivity
- First request after sleep may take 10-30 seconds
- Consider upgrading to a paid plan for production use

### Debugging
- Check the logs in Render dashboard if deployment fails
- Common issues:
  - Missing dependencies in requirements.txt
  - Port binding issues (make sure to use `$PORT`)
  - Missing environment variables

## Local Development

To run locally with production-like settings:

```bash
export GEMINI_API_KEY="your-api-key"
export FLASK_ENV="production"
export PORT="5000"
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT app_web:app
```

## Support

If you encounter issues:
1. Check Render logs for error messages
2. Ensure all environment variables are set correctly
3. Verify that your Gemini API key is valid and has access to the Gemini 2.0 Flash model
4. Check that all dependencies are in requirements.txt