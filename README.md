# PaperSurf 🏄‍♂️📄

An AI-powered daily academic paper recommendation system that helps researchers stay up-to-date with the latest arXiv publications. PaperSurf fetches daily papers from arXiv, uses topic modeling to calculate similarity with papers from your Zotero library, and generates personalized email recommendations to help you keep pace with the daily paper waves. 🌊

## Features ✨

- 📅 **Daily arXiv Monitoring**: Automatically fetches new papers from specified arXiv categories
- 🧠 **Intelligent Matching**: Uses embedding models to calculate similarity between new papers and your research interests
- 📚 **Multi-Directory Zotero Integration**: Analyzes papers from multiple Zotero directories to understand different research interests and generate targeted recommendations
- 📧 **Email Notifications**: Generates HTML email digests with top paper recommendations
- ⚙️ **Configurable**: Customize categories, models, and recommendation parameters

## Setup 🚀

### Option 1: GitHub Actions (Recommended - No Coding, Free, No Local Setup ) ☁️

The easiest way to use PaperSurf is through GitHub Actions, which runs automatically in the cloud thanks to Github:

1. **Fork and Star this repository** to your GitHub account

2. **Configure secrets** in your repository settings (`Settings > Secrets and variables > Actions`):
   - `EMAIL_SENDER`: Your Gmail address
   - `EMAIL_RECEIVER`: Email address to receive recommendations
   - `EMAIL_PASSWORD`: Gmail app-specific password
   - `ZOTERO_API_KEY`: Your Zotero API key
   - `ZOTERO_LIBRARY_ID`: Your Zotero library ID

3. **Customize configuration** by editing `config.yaml` in your fork

4. **Enable GitHub Actions** - the workflow will run daily automatically

### Option 2: Local Installation 💻

For local development or manual runs:

#### Prerequisites
- Python 3.8+
- uv (for dependency management)
- Zotero account with API access

#### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd paper-surf
   ```

2. **Install dependencies:**
   ```bash
   uv venv venv
   source venv/bin/activate
   uv pip install .
   ```

3. **Configure environment variables:**
   ```bash
   # Copy and edit the setup script
   cp setup_envs.sh setup_envs_local.sh
   # Edit setup_envs_local.sh with your credentials
   source setup_envs_local.sh
   ```

4. **Configure the system:**
   Edit `config.yaml` to customize (see [Configuration](#configuration-️) section for details):
   - arXiv categories and search parameters
   - Zotero collections and directories
   - Embedding model settings
   - Email notification preferences
   - Output and digest formatting

### Usage 🎯

Run the daily recommendation system:
```bash
python run_daily.py
```

With custom parameters:
```bash
python run_daily.py --date 2025-07-21 --topk 10
```

## Project Structure 📁

```
.
├── app/                # Application modules
│   ├── fetcher.py      # arXiv paper fetching
│   ├── embedder.py     # Text embedding and similarity
│   ├── recommender.py  # Recommendation engine
│   └── notifier.py     # Email notification
├── data/               # Data files (embeddings, papers)
├── output/             # Generated HTML digests
├── config.yaml         # Configuration file
├── run_daily.py        # Main application script
└── pyproject.toml      # Project dependencies
```

## Configuration ⚙️

### Zotero Setup 📚
1. Get your Zotero API key from https://www.zotero.org/settings/keys
2. Find your library ID from your Zotero profile
3. Set these in your environment variables

### Email Setup 📧
Configure SMTP settings for email notifications:
- Email sender and receiver addresses
- App-specific password for Gmail or other providers

### Configuration File (`config.yaml`) 📄

The `config.yaml` file contains all system settings:

#### arXiv Settings
```yaml
arxiv:
  categories: ["cs.CL", "cs.LG", "cs.AI", "stat.ML", "cs.CV", "cs.IR"]
  lookback_days: 2          # How many days back to search for papers
  max_results: 10           # Maximum papers to fetch per category
```

#### Interest Papers & Zotero Collections
```yaml
interest_papers:
  zotero_collections:
    deep_research: "0-PhD-LLM/9-agent-robust/deep-research"
    # Add more collections for different research areas
    # nlp_research: "path/to/nlp/collection"
    # cv_research: "path/to/cv/collection"
```

#### Model & Recommendation Settings
```yaml
embedding_model: "allenai/specter2_base"  # Embedding model for similarity
top_k: 5                                  # Number of top recommendations
use_seen_papers: false                    # Track previously seen papers
```

#### Output & Digest Settings
```yaml
digest:
  title: "PaperSurf Daily Digest"
  subtitle: "AI & Machine Learning Paper Recommendations"
  output_dir: "output"
  open_in_browser: false                  # Auto-open digest after generation

output_path: "output/digest.html"
```

#### Pipeline Control
```yaml
pipeline:
  skip_fetch: false                       # Skip fetching new papers (for testing)
  skip_embed: false                       # Skip embedding generation (for testing)
```

#### Notification Settings
```yaml
notification:
  method: email
  auto_send: true                         # Automatically send email notifications
  dry_run: false                          # Test mode without sending emails
  email:
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
```

## How It Works 🔄

1. 📥 **Fetch**: Downloads new papers from specified arXiv categories
2. 🔤 **Embed**: Generates embeddings for paper abstracts using transformer models
3. 🔍 **Match**: Calculates similarity between new papers and your Zotero directories
4. 📊 **Rank**: Scores and ranks papers based on relevance to your interests
5. 📨 **Notify**: Sends HTML email digest with top recommendations
