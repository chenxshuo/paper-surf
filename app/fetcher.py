#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fetcher.py
Module for fetching recent papers from arXiv API based on configuration.
"""

import arxiv
import json
import os
from datetime import datetime, timedelta
import logging
from app.models import Paper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('papersurf.fetcher')

def fetch_papers(config, override_date=None):
    """
    Fetch papers from arXiv API based on configuration
    
    Args:
        config (dict): Configuration dictionary
        override_date (str, optional): Override date in YYYY-MM-DD format
        
    Returns:
        list: List of Paper objects
    """
    # Get arXiv configuration
    categories = config['arxiv']['categories']
    lookback_days = config['arxiv']['lookback_days']
    
    # Determine the date range
    if override_date:
        target_date = datetime.strptime(override_date, '%Y-%m-%d')
    else:
        target_date = datetime.now()
    
    # Using a simple natural day-based approach for lookback
    # Note: This doesn't account for arXiv's specific submission schedule,
    # but provides a straightforward way to fetch papers from the last N days
    
    # arXiv uses date format YYYYMMDDHHMMSS
    date_until_str = target_date.strftime('%Y%m%d235959')
    date_from_str = (target_date - timedelta(days=lookback_days)).strftime('%Y%m%d000000')
    
    logger.info(f"Fetching papers from {date_from_str} to {date_until_str} for categories: {categories}")
    
    # Build the search query
    category_query = ' OR '.join([f'cat:{cat}' for cat in categories])
    query = f"({category_query}) AND submittedDate:[{date_from_str} TO {date_until_str}]"
    
    # Set up the arXiv client
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0,  # Be nice to the API
        num_retries=3
    )
    
    # Create the search
    search = arxiv.Search(
        query=query,
        max_results=config['arxiv']['max_results'],  # Reasonable limit for daily papers
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    # Fetch results
    try:
        results = list(client.results(search))
        logger.info(f"Successfully fetched {len(results)} papers")
    except Exception as e:
        logger.error(f"Error fetching papers: {e}")
        return []
    
    # Convert to Paper objects
    papers = [Paper.from_arxiv_result(result) for result in results]
    logger.info(f"Converted {len(papers)} results to Paper objects")
    
    return papers

def save_candidate_papers(papers, output_dir='data'):
    """
    Save fetched papers to a JSONL file
    
    Args:
        papers (list): List of Paper objects
        output_dir (str): Directory to save the output file
        
    Returns:
        str: Path to the saved file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename with current date
    date_str = datetime.now().strftime('%Y-%m-%d')
    output_file = os.path.join(output_dir, f'candidate_papers_{date_str}.jsonl')
    
    # Write papers to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for paper in papers:
            f.write(paper.to_json() + '\n')
    
    logger.info(f"Saved {len(papers)} candidate papers to {output_file}")
    return output_file



if __name__ == "__main__":
    # For testing purposes
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    papers = fetch_papers(config)
    save_candidate_papers(papers)
