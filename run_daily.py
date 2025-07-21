#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_daily.py
Main script for PaperSurf that orchestrates the daily paper recommendation pipeline.
"""

import os
import yaml
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv
from app.fetcher import fetch_papers, save_candidate_papers
from app.reader import load_interest_papers
from app.embedder import embed_papers, save_embeddings
from app.recommender import get_recommendations
from app.formatter import format_html_digest, save_html_digest
from app.notifier import send_notification

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('papersurf.main')

def load_config():
    """
    Load configuration from config.yaml
    
    Returns:
        dict: Configuration parameters
    """
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def main():
    """
    Main function to run the daily paper recommendation pipeline
    """
    # Load configuration first
    config = load_config()
    logger.info("Configuration loaded successfully")
    
    # Parse command line arguments (only for overriding config)
    parser = argparse.ArgumentParser(description='PaperSurf: Daily Academic Paper Recommender')
    parser.add_argument('--lookback-days', type=int, help='Override lookback days')
    parser.add_argument('--date', type=str, help='Override date (YYYY-MM-DD)')
    parser.add_argument('--topk', type=int, help='Override number of recommendations')
    parser.add_argument('--skip-fetch', action='store_true', help='Override: Skip fetching new papers')
    parser.add_argument('--skip-embed', action='store_true', help='Override: Skip embedding papers')
    parser.add_argument('--open-digest', action='store_true', help='Override: Open digest in browser after generation')
    parser.add_argument('--no-email', action='store_true', help='Override: Disable email notification')
    parser.add_argument('--dry-run', action='store_true', help='Override: Run without sending notifications')
    parser.add_argument('--use-seen-papers', type=bool, nargs='?', const=True, help='Override: Enable/disable tracking of seen papers')
    args = parser.parse_args()
    
    # Apply command-line overrides to config if specified
    if args.lookback_days:
        config['arxiv']['lookback_days'] = args.lookback_days
        logger.info(f"Overriding lookback_days to {args.lookback_days}")
    if args.topk:
        config['top_k'] = args.topk
        logger.info(f"Overriding top_k to {args.topk}")
    
    if args.skip_fetch:
        config['pipeline']['skip_fetch'] = True
        logger.info("Overriding: Skipping paper fetching")
        
    if args.skip_embed:
        config['pipeline']['skip_embed'] = True
        logger.info("Overriding: Skipping paper embedding")
        
    if args.open_digest:
        config['digest']['open_in_browser'] = True
        logger.info("Overriding: Will open digest in browser")
        
    if args.no_email:
        config['notification']['auto_send'] = False
        logger.info("Overriding: Email notifications disabled")
        
    if args.dry_run:
        config['notification']['dry_run'] = True
        logger.info("Overriding: Running in dry-run mode")
        
    if args.use_seen_papers is not None:
        config['use_seen_papers'] = args.use_seen_papers
        logger.info(f"Overriding: Seen papers tracking {'enabled' if args.use_seen_papers else 'disabled'}")
    
    # Get date string
    date_str = args.date if args.date else datetime.now().strftime('%Y-%m-%d')
    
    # Initialize papers variable
    papers = []
    
    # Initialize total papers examined counter
    total_papers_examined = 0
    
    # 1. Fetch daily papers
    if not config['pipeline'].get('skip_fetch', False):
        logger.info("Step 1: Fetching papers from arXiv")
        papers = fetch_papers(config, override_date=args.date)
        if papers:
            total_papers_examined = len(papers)
            candidate_file = save_candidate_papers(papers)
            logger.info(f"Saved {len(papers)} candidate papers to {candidate_file}")
        else:
            logger.warning("No papers fetched, stopping pipeline")
            return
    else:
        logger.info("Skipping paper fetching step")
        # Load papers from the most recent candidate file
        data_dir = 'data'
        candidate_files = [f for f in os.listdir(data_dir) if f.startswith('candidate_papers_') and f.endswith('.jsonl')]
        if candidate_files:
            # Get the most recent file
            latest_file = sorted(candidate_files)[-1]
            logger.info(f"Loading papers from {latest_file}")
            import json
            from app.models import Paper
            with open(os.path.join(data_dir, latest_file), 'r') as f:
                papers = []
                for line in f:
                    if line.strip():
                        paper_dict = json.loads(line)
                        papers.append(Paper.from_dict(paper_dict))
                total_papers_examined = len(papers)
                logger.info(f"Loaded {len(papers)} papers from file")
        else:
            logger.warning("No candidate paper files found, stopping pipeline")
            return
    
    # 2. Load interest papers
    logger.info("Step 2: Loading interest papers")
    interest_papers = load_interest_papers(config)
    if interest_papers:
        total_papers = sum(len(papers_list) for papers_list in interest_papers.values())
        logger.info(f"Loaded {total_papers} interest papers")
        for source_type, papers_list in interest_papers.items():
            logger.info(f"  - {source_type}: {len(papers_list)} papers")
    else:
        logger.warning("No interest papers found, stopping pipeline")
        return
    
    # 3. Embed papers
    if not config['pipeline'].get('skip_embed', False):
        logger.info("Step 3: Embedding papers")
        
        # Embed candidate papers
        logger.info("Embedding candidate papers...")
        candidate_embeddings = embed_papers(papers, config)
        if candidate_embeddings:
            save_embeddings(candidate_embeddings, 'data/candidate_embeddings.npz')
            logger.info(f"Embedded {len(candidate_embeddings)} candidate papers")
        else:
            logger.warning("Failed to embed candidate papers, stopping pipeline")
            return
        
        # Embed interest papers
        logger.info("Embedding interest papers...")
        all_interest_papers = []
        for papers_list in interest_papers.values():
            all_interest_papers.extend(papers_list)
        
        interest_embeddings = embed_papers(all_interest_papers, config)
        if interest_embeddings:
            save_embeddings(interest_embeddings, 'data/interest_embeddings.npz')
            logger.info(f"Embedded {len(interest_embeddings)} interest papers")
        else:
            logger.warning("Failed to embed interest papers, stopping pipeline")
            return
    else:
        logger.info("Skipping embedding step")
    
    # 4. Get recommendations
    logger.info("Step 4: Generating recommendations")
    use_seen_papers = config.get('use_seen_papers', False)
    logger.info(f"Seen papers tracking is {'enabled' if use_seen_papers else 'disabled'}")    
    # Create a dictionary mapping paper IDs to papers for reference
    all_interest_papers_dict = {}
    for category, papers_list in interest_papers.items():
        for paper in papers_list:
            all_interest_papers_dict[paper.id] = paper
    
    recommendations = get_recommendations(config, papers, interest_papers, use_seen_papers=use_seen_papers)
    
    if not recommendations:
        logger.warning("No recommendations generated")
        return
    
    # Print recommendations to console
    total_recs = sum(len(recs) for recs in recommendations.values())
    logger.info(f"Generated {total_recs} total recommendations")
    
    print("\n===== Today's Paper Recommendations =====\n")
    
    # Print recommendations based on my publications
    if 'my_pubs' in recommendations and recommendations['my_pubs']:
        print("\n--- Based on My Publications ---\n")
        for i, (paper, score, ref_paper) in enumerate(recommendations['my_pubs'], 1):
            print(f"{i}. {paper.title} (Score: {score:.4f})")
            print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
            print(f"   URL: {paper.url}")
            print(f"   Abstract: {paper.abstract[:200]}...")
            print()
    
    # Print recommendations for each research direction
    direction_keys = [key for key in recommendations.keys() if key.startswith('zotero_collections.')]
    for key in direction_keys:
        direction_name = key.split('.', 1)[1]  # Extract name after the dot
        if recommendations[key]:
            print(f"\n--- Based on {direction_name.replace('_', ' ').title()} ---\n")
            for i, (paper, score, ref_paper) in enumerate(recommendations[key], 1):
                print(f"{i}. {paper.title} (Score: {score:.4f})")
                print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
                print(f"   URL: {paper.url}")
                print(f"   Abstract: {paper.abstract[:200]}...")
                print()
    
    # Print legacy interesting directions if they exist
    if 'interesting_directions' in recommendations and recommendations['interesting_directions']:
        print("\n--- Based on Interesting Directions (Legacy) ---\n")
        for i, (paper, score, ref_paper) in enumerate(recommendations['interesting_directions'], 1):
            print(f"{i}. {paper.title} (Score: {score:.4f})")
            print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
            print(f"   URL: {paper.url}")
            print(f"   Abstract: {paper.abstract[:200]}...")
            print()
    
    # 5. Generate digest
    logger.info("Step 5: Generating HTML digest")
    logger.info(f"Total papers examined: {total_papers_examined}")
    html_content = format_html_digest(recommendations, config, date_str, total_papers_examined=total_papers_examined)
    if html_content:
        # Save HTML digest
        filepath = save_html_digest(html_content)
        if filepath:
            logger.info(f"HTML digest saved to {filepath}")
            
            # Open in browser if configured
            if config['digest'].get('open_in_browser', False):
                import subprocess
                subprocess.run(['open', filepath])
                logger.info(f"Opened digest in browser")
    else:
        logger.warning("Failed to generate HTML digest")
        return
    
    # 6. Send notification
    if not config['notification'].get('dry_run', False) and config['notification'].get('auto_send', False):
        logger.info("Step 6: Sending notification")
        if send_notification(html_content, config, date_str):
            logger.info("Notification sent successfully")
        else:
            logger.error("Failed to send notification")
    else:
        logger.info("Skipping notification step (dry run or auto-send disabled)")
    
    logger.info("PaperSurf pipeline completed successfully!")

if __name__ == "__main__":
    main()
