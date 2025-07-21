# PaperSurf

A lightweight and extensible daily academic paper recommender system.

## Docker Development Setup

### Prerequisites
- Docker
- Docker Compose

### Getting Started

1. **Build the Docker image:**
   ```bash
   docker-compose build
   ```

2. **Run the development environment:**
   ```bash
   docker-compose run dev
   ```
   This will give you an interactive shell inside the container where you can run commands.

3. **Run the PaperSurf application:**
   ```bash
   docker-compose run papersurf
   ```
   
   With custom arguments:
   ```bash
   docker-compose run papersurf python run_daily.py --date 2025-06-06 --topk 5
   ```

### Project Structure

```
.
├── app/                # Application modules
├── data/               # Data files (interest papers, embeddings)
├── output/             # Generated HTML digests
├── config.yaml         # Configuration file
├── run_daily.py        # Main script
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
└── requirements.txt    # Python dependencies
```

## Development Workflow

1. Make changes to the code
2. Run the application inside Docker to test changes
3. Repeat

## Configuration

Edit `config.yaml` to customize:
- arXiv categories to monitor
- Interest papers location
- Embedding model
- Email notification settings

### Proxy Configuration

PaperSurf now supports optional proxy configuration for Docker:

1. **Enable/disable proxy usage:**
   ```bash
   # Enable proxy
   export USE_PROXY=true
   
   # Disable proxy (default)
   export USE_PROXY=false
   ```

2. **Set proxy environment variables if needed:**
   ```bash
   export HTTP_PROXY=http://your-proxy-server:port
   export HTTPS_PROXY=http://your-proxy-server:port
   export NO_PROXY=localhost,127.0.0.1
   ```

3. **Build with proxy settings:**
   ```bash
   # With proxy enabled
   USE_PROXY=true docker-compose build
   
   # Without proxy (default)
   docker-compose build
   ```

These settings affect both the Docker build process and the runtime environment inside containers.
