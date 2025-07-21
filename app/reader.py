#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reader.py
Module for loading interest papers from various formats (BibTeX, JSONL).
"""

import json
import os
import re
import logging
import bibtexparser
from bibtexparser.customization import convert_to_unicode
from typing import List, Dict, Any, Optional, Union, Tuple
from app.models import Paper
from pyzotero import zotero

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('papersurf.reader')

def extract_arxiv_id(url: str) -> Optional[str]:
    """
    Extract arXiv ID from a URL or string.
    
    Args:
        url: URL or string containing arXiv ID
        
    Returns:
        str: arXiv ID or None if not found
    """
    if not url:
        return None
        
    # Try to extract from URL
    arxiv_pattern = r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)'
    match = re.search(arxiv_pattern, url)
    if match:
        return match.group(1)
    
    # Try to extract from arXiv:XXXX.XXXXX format
    arxiv_id_pattern = r'arxiv:([0-9]+\.[0-9]+)'
    match = re.search(arxiv_id_pattern, url, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return None

def load_bib_papers(filepath: str, source_type: str = 'user_interest') -> List[Paper]:
    """
    Load papers from a BibTeX file.
    
    Args:
        filepath: Path to the BibTeX file
        source_type: Type of interest papers (e.g., 'my_pubs', 'interesting_directions.indirect_prompt_injection')
        
    Returns:
        List[Paper]: List of Paper objects
    """
    if not os.path.exists(filepath):
        logger.error(f"BibTeX file not found: {filepath}")
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as bibtex_file:
            parser = bibtexparser.bparser.BibTexParser(
                customization=convert_to_unicode
            )
            bib_database = bibtexparser.load(bibtex_file, parser=parser)
        
        papers = []
        for entry in bib_database.entries:
            # Extract arXiv ID if available
            arxiv_id = None
            if 'url' in entry:
                arxiv_id = extract_arxiv_id(entry['url'])
            elif 'journal' in entry and 'arxiv' in entry['journal'].lower():
                arxiv_id = extract_arxiv_id(entry['journal'])
            
            # Use entry ID if no arXiv ID found
            paper_id = arxiv_id or entry.get('ID', '')
            
            # Create Paper object
            paper = Paper(
                id=paper_id,
                title=entry.get('title', '').replace('{', '').replace('}', ''),
                abstract=entry.get('abstract', ''),
                authors=[a.strip() for a in entry.get('author', '').split(' and ')],
                categories=[],  # BibTeX doesn't typically include categories
                published=entry.get('year', ''),
                url=entry.get('url', None),
                source=source_type,
                metadata={
                    'bibtex_id': entry.get('ID', ''),
                    'journal': entry.get('journal', ''),
                    'year': entry.get('year', '')
                }
            )
            papers.append(paper)
        
        logger.info(f"Loaded {len(papers)} papers from BibTeX file: {filepath}")
        return papers
    
    except Exception as e:
        logger.error(f"Error loading BibTeX file: {e}")
        return []

def convert_zotero_to_paper(zotero_item: dict, source_type: str) -> Paper:
    """
    Convert a Zotero item to a Paper object.
    
    Args:
        zotero_item: Zotero item dictionary
        source_type: Type of interest papers (e.g., 'zotero_collections.indirect_prompt_injection')
        
    Returns:
        Paper: Paper object
    """
    # Extract data from Zotero item
    data = zotero_item['data']
    
    # Extract arXiv ID if available in URL or DOI
    arxiv_id = None
    if 'url' in data:
        arxiv_id = extract_arxiv_id(data['url'])
    elif 'DOI' in data:
        arxiv_id = extract_arxiv_id(data['DOI'])
    
    # Use item key if no arXiv ID found
    paper_id = arxiv_id or zotero_item['key']
    
    # Extract authors
    authors = []
    if 'creators' in data:
        for creator in data['creators']:
            if creator['creatorType'] == 'author':
                if 'name' in creator:
                    authors.append(creator['name'])
                else:
                    name = f"{creator.get('lastName', '')}, {creator.get('firstName', '')}"
                    authors.append(name.strip(', '))
    
    # Create Paper object
    paper = Paper(
        id=paper_id,
        title=data.get('title', ''),
        abstract=data.get('abstractNote', ''),
        authors=authors,
        categories=[],  # Zotero doesn't typically include categories like arXiv
        published=data.get('date', ''),
        url=data.get('url', None),
        source=source_type,
        metadata={
            'zotero_key': zotero_item['key'],
            'item_type': data.get('itemType', ''),
            'publication': data.get('publicationTitle', ''),
            'doi': data.get('DOI', '')
        }
    )
    return paper


def get_zotero_corpus_by_collections(library_id: str, api_key: str, desired_collections: List[str] = None) -> Dict[str, List[dict]]:
    """
    Get Zotero corpus organized by collection names.
    
    Args:
        library_id: Zotero library ID
        api_key: Zotero API key
        desired_collections: Optional list of collection paths to filter by. If provided,
                            only papers from these collections will be included.
        
    Returns:
        Dictionary where keys are collection names and values are lists of papers in those collections
    """
    zot = zotero.Zotero(library_id, 'user', api_key)
    # Get all collections
    collections = zot.everything(zot.collections())
    collections_dict = {c['key']: c for c in collections}
    
    # Get all papers (conference papers, journal articles, and preprints)
    corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    # Filter out papers without abstracts
    corpus = [c for c in corpus if c['data']['abstractNote'] != '']
    
    # Helper function to get collection path
    def get_collection_path(col_key: str) -> str:
        if p := collections_dict[col_key]['data']['parentCollection']:
            return get_collection_path(p) + '/' + collections_dict[col_key]['data']['name']
        else:
            return collections_dict[col_key]['data']['name']
    
    # Organize papers by collection
    result = {}
    for paper in corpus:
        # Get collection paths for this paper
        collection_keys = paper['data']['collections']
        if not collection_keys:  # Skip papers not in any collection
            continue
            
        for col_key in collection_keys:
            try:
                collection_path = get_collection_path(col_key)
                # Skip if we only want specific collections and this isn't one of them
                if desired_collections and collection_path not in desired_collections:
                    continue
                    
                if collection_path not in result:
                    result[collection_path] = []
                result[collection_path].append(paper)
            except KeyError:
                # Skip if collection key is not found
                continue
    
    return result


def load_jsonl_papers(filepath: str, source_type: str = 'user_interest') -> List[Paper]:
    """
    Load papers from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        source_type: Type of interest papers (e.g., 'my_pubs', 'interesting_directions.indirect_prompt_injection')
        
    Returns:
        List[Paper]: List of Paper objects
    """
    if not os.path.exists(filepath):
        logger.error(f"JSONL file not found: {filepath}")
        return []
    
    try:
        papers = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    paper_dict = json.loads(line)
                    paper = Paper.from_dict(paper_dict)
                    paper.source = source_type  # Set source type
                    papers.append(paper)
        
        logger.info(f"Loaded {len(papers)} papers from JSONL file: {filepath}")
        return papers
    
    except Exception as e:
        logger.error(f"Error loading JSONL file: {e}")
        return []

def load_interest_papers(config: Dict[str, Any]) -> Dict[str, List[Paper]]:
    """
    Load interest papers from configured sources.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict[str, List[Paper]]: Dictionary with paper types as keys and lists of Paper objects as values
    """
    interest_papers = {
        'my_pubs': []
    }
    
    # Check if we have the new config format
    if 'interest_papers' in config:
        # Load my publications
        if 'my_pubs' in config['interest_papers']:
            my_pubs_path = config['interest_papers']['my_pubs']
            if my_pubs_path.endswith('.bib'):
                interest_papers['my_pubs'] = load_bib_papers(my_pubs_path, 'my_pubs')
            elif my_pubs_path.endswith('.jsonl'):
                interest_papers['my_pubs'] = load_jsonl_papers(my_pubs_path, 'my_pubs')
            
            logger.info(f"Loaded {len(interest_papers['my_pubs'])} papers from my publications")
        
        # Load interesting directions - now handling as a dictionary
        if 'interesting_directions' in config['interest_papers']:
            directions_config = config['interest_papers']['interesting_directions']
            
            # Handle the case where interesting_directions is a dictionary
            if isinstance(directions_config, dict):
                for direction_name, path in directions_config.items():
                    source_type = f"interesting_directions.{direction_name}"
                    
                    if path.endswith('.bib'):
                        papers = load_bib_papers(path, source_type)
                    elif path.endswith('.jsonl'):
                        papers = load_jsonl_papers(path, source_type)
                    else:
                        logger.warning(f"Unsupported file format for {path}")
                        continue
                    
                    # Add this direction to the interest_papers dictionary
                    interest_papers[source_type] = papers
                    logger.info(f"Loaded {len(papers)} papers from {direction_name}")
            
            # Handle legacy list format for backward compatibility
            elif isinstance(directions_config, list):
                # Create a combined list for all directions
                combined_directions = []
                
                for path in directions_config:
                    if path.endswith('.bib'):
                        papers = load_bib_papers(path, 'interesting_directions')
                    elif path.endswith('.jsonl'):
                        papers = load_jsonl_papers(path, 'interesting_directions')
                    else:
                        logger.warning(f"Unsupported file format for {path}")
                        continue
                    
                    combined_directions.extend(papers)
                
                interest_papers['interesting_directions'] = combined_directions
                logger.info(f"Loaded {len(combined_directions)} papers from interesting directions (legacy format)")

        if 'zotero_collections' in config['interest_papers']:
            # Check if we have Zotero API credentials
            zotero_api_key = os.environ.get('ZOTERO_API_KEY')
            zotero_library_id = os.environ.get('ZOTERO_LIBRARY_ID')

            # Get collection mappings from config
            collection_mappings = config['interest_papers']['zotero_collections']
            
            try:
                # Get papers from Zotero collections
                collection_paths = list(collection_mappings.values())
                zotero_collections = get_zotero_corpus_by_collections(zotero_library_id, zotero_api_key, collection_paths)
                
                # Process each collection
                for direction_name, collection_path in collection_mappings.items():
                    if collection_path in zotero_collections:
                        # Convert Zotero items to Paper objects
                        source_type = f"zotero_collections.{direction_name}"
                        papers = [convert_zotero_to_paper(item, source_type) for item in zotero_collections[collection_path]]
                        
                        # Add to interest papers
                        interest_papers[source_type] = papers
                        logger.info(f"Loaded {len(papers)} papers from Zotero collection: {direction_name} ({collection_path})")
                    else:
                        logger.warning(f"No papers found in Zotero collection: {collection_path}")
            except Exception as e:
                logger.error(f"Error loading papers from Zotero: {e}")


    # Fallback to old config format if no papers were loaded
    if not any(interest_papers.values()) and 'interest_papers_path' in config:
        logger.info("Using legacy interest_papers_path configuration")
        interest_path = config['interest_papers_path']
        
        if interest_path.endswith('.bib'):
            interest_papers['my_pubs'] = load_bib_papers(interest_path)
        elif interest_path.endswith('.jsonl'):
            interest_papers['my_pubs'] = load_jsonl_papers(interest_path)
    
    total_papers = sum(len(papers) for papers in interest_papers.values())
    logger.info(f"Total interest papers loaded: {total_papers}")
    return interest_papers

if __name__ == "__main__":
    # For testing purposes
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    from dotenv import load_dotenv
    load_dotenv()
    # Load interest papers
    interest_papers = load_interest_papers(config)
    # Print loaded papers
    print("\n=== Interest Papers ===\n")
    for category, papers in interest_papers.items():
        print(f"\n--- {category} ({len(papers)} papers) ---\n")
        for i, paper in enumerate(papers[:3], 1):  # Show first 3 papers
            print(f"{i}. {paper.title}")
            print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
            print(f"   Source: {paper.source}")
            print()
