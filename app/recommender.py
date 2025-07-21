#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
recommender.py
Module for recommending papers based on embeddings and similarity.
"""

import os
import json
import numpy as np
import logging
from typing import List, Dict, Any, Set, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from app.models import Paper
from app.embedder import load_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('papersurf.recommender')

def load_seen_papers(filepath: str = 'data/seen.json', use_seen_papers: bool = False) -> Set[str]:
    """
    Load the set of paper IDs that have already been seen/recommended.
    
    Args:
        filepath: Path to the JSON file containing seen paper IDs
        use_seen_papers: Whether to use the seen papers functionality (default: False)
        
    Returns:
        Set[str]: Set of seen paper IDs
    """
    if not use_seen_papers:
        logger.info("Seen papers functionality is disabled")
        return set()
        
    if not os.path.exists(filepath):
        logger.info(f"No seen papers file found at {filepath}, creating new one")
        return set()
    
    try:
        with open(filepath, 'r') as f:
            seen_ids = set(json.load(f))
        logger.info(f"Loaded {len(seen_ids)} seen paper IDs")
        return seen_ids
    except Exception as e:
        logger.error(f"Error loading seen papers: {e}")
        return set()

def save_seen_papers(seen_ids: Set[str], filepath: str = 'data/seen.json', use_seen_papers: bool = False):
    """
    Save the set of seen paper IDs to a JSON file.
    
    Args:
        seen_ids: Set of paper IDs that have been seen
        filepath: Path to save the JSON file
        use_seen_papers: Whether to use the seen papers functionality (default: False)
    """
    if not use_seen_papers:
        logger.debug("Seen papers functionality is disabled, not saving seen papers")
        return
        
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(list(seen_ids), f)
        logger.info(f"Saved {len(seen_ids)} seen paper IDs to {filepath}")
    except Exception as e:
        logger.error(f"Error saving seen papers: {e}")

def compute_similarity_scores(
    candidate_embeddings: Dict[str, np.ndarray],
    interest_embeddings: Dict[str, np.ndarray]
) -> Dict[str, Tuple[float, str]]:
    """
    Compute similarity scores between candidate papers and interest papers.
    
    Args:
        candidate_embeddings: Dictionary mapping candidate paper IDs to embeddings
        interest_embeddings: Dictionary mapping interest paper IDs to embeddings
        
    Returns:
        Dict[str, float]: Dictionary mapping candidate paper IDs to similarity scores
    """
    if not candidate_embeddings or not interest_embeddings:
        logger.warning("Empty embeddings provided for similarity computation")
        return {}
    
    # Convert embeddings to matrices
    candidate_ids = list(candidate_embeddings.keys())
    interest_ids = list(interest_embeddings.keys())
    
    candidate_matrix = np.array([candidate_embeddings[id_] for id_ in candidate_ids])
    interest_matrix = np.array([interest_embeddings[id_] for id_ in interest_ids])
    
    # Compute cosine similarity
    try:
        # Compute similarity between each candidate and all interest papers
        similarity_matrix = cosine_similarity(candidate_matrix, interest_matrix)
        
        # For each candidate, find the maximum similarity and the index of the most similar interest paper
        max_similarities = np.max(similarity_matrix, axis=1)
        most_similar_indices = np.argmax(similarity_matrix, axis=1)
        
        # Create a dictionary mapping candidate IDs to (score, most_similar_interest_id) tuples
        scores = {}
        for i, candidate_id in enumerate(candidate_ids):
            most_similar_idx = most_similar_indices[i]
            most_similar_interest_id = interest_ids[most_similar_idx]
            scores[candidate_id] = (float(max_similarities[i]), most_similar_interest_id)
        
        logger.info(f"Computed similarity scores for {len(scores)} candidate papers")
        return scores
    
    except Exception as e:
        logger.error(f"Error computing similarity scores: {e}")
        return {}

def recommend_papers(
    papers: List[Paper],
    candidate_embeddings: Dict[str, np.ndarray],
    interest_embeddings: Dict[str, np.ndarray],
    interest_papers: Dict[str, Paper] = None,
    top_k: int = 10,
    seen_ids: Set[str] = None,
    use_seen_papers: bool = False
) -> List[Tuple[Paper, float, Paper]]:
    """
    Recommend papers based on similarity to interest papers.
    
    Args:
        papers: List of candidate Paper objects
        candidate_embeddings: Dictionary mapping candidate paper IDs to embeddings
        interest_embeddings: Dictionary mapping interest paper IDs to embeddings
        top_k: Number of papers to recommend
        seen_ids: Set of already seen paper IDs to exclude
        
    Returns:
        List[Tuple[Paper, float]]: List of (paper, score) tuples for recommended papers
    """
    if seen_ids is None:
        seen_ids = load_seen_papers()
    
    # Compute similarity scores
    scores = compute_similarity_scores(candidate_embeddings, interest_embeddings)
    
    # Filter out seen papers and create (paper, score, reference_paper) tuples
    paper_dict = {paper.id: paper for paper in papers}
    interest_paper_dict = interest_papers or {}
    paper_scores = []
    
    for paper_id, (score, ref_paper_id) in scores.items():
        if paper_id not in seen_ids and paper_id in paper_dict:
            # Find the reference paper that this recommendation is based on
            reference_paper = None
            if interest_paper_dict and ref_paper_id in interest_paper_dict:
                reference_paper = interest_paper_dict[ref_paper_id]
            paper_scores.append((paper_dict[paper_id], score, reference_paper))
    
    # Sort by score in descending order and take top-k
    paper_scores.sort(key=lambda x: x[1], reverse=True)
    recommendations = paper_scores[:top_k]
    
    logger.info(f"Generated {len(recommendations)} recommendations out of {len(papers)} candidates")
    
    # Update seen papers only if tracking is enabled
    if use_seen_papers:
        for paper, _, _ in recommendations:
            seen_ids.add(paper.id)
        save_seen_papers(seen_ids)
        logger.info(f"Updated seen papers tracking with {len(recommendations)} new papers")
    else:
        logger.info("Seen papers tracking is disabled, not updating seen papers")
    
    return recommendations

def filter_embeddings_by_source(
    papers: List[Paper],
    embeddings: Dict[str, np.ndarray],
    source_type: str
) -> Dict[str, np.ndarray]:
    """
    Filter embeddings to only include those from papers with the specified source type.
    
    Args:
        papers: List of Paper objects
        embeddings: Dictionary mapping paper IDs to embeddings
        source_type: Source type to filter by (e.g., 'my_pubs', 'interesting_directions.indirect_prompt_injection')
        
    Returns:
        Dict[str, np.ndarray]: Filtered embeddings dictionary
    """
    filtered_embeddings = {}
    source_papers = [p for p in papers if p.source == source_type]
    source_ids = {p.id for p in source_papers}
    
    for paper_id, embedding in embeddings.items():
        if paper_id in source_ids:
            filtered_embeddings[paper_id] = embedding
    
    logger.info(f"Filtered {len(filtered_embeddings)} embeddings for source type: {source_type}")
    return filtered_embeddings

def get_recommendations_by_type(
    config: Dict[str, Any],
    papers: List[Paper],
    interest_papers: Dict[str, List[Paper]],
    candidate_embeddings: Dict[str, np.ndarray],
    interest_embeddings: Dict[str, np.ndarray],
    source_type: str,
    top_k: int,
    seen_ids: Set[str],
    use_seen_papers: bool = False
) -> List[Tuple[Paper, float, Paper]]:
    """
    Get recommendations based on a specific type of interest papers.
    
    Args:
        config: Configuration dictionary
        papers: List of candidate papers
        interest_papers: Dictionary of interest papers by type
        candidate_embeddings: Dictionary mapping candidate paper IDs to embeddings
        interest_embeddings: Dictionary mapping interest paper IDs to embeddings
        source_type: Type of interest papers to use (e.g., 'my_pubs', 'interesting_directions.indirect_prompt_injection')
        top_k: Number of papers to recommend
        seen_ids: Set of already seen paper IDs
        
    Returns:
        List[Tuple[Paper, float]]: List of (paper, score) tuples for recommended papers
    """
    # Filter interest embeddings to only include those from the specified source type
    if source_type in interest_papers and interest_papers[source_type]:
        # Filter embeddings to only include those from the specified source type
        filtered_interest_embeddings = {}
        for paper in interest_papers[source_type]:
            if paper.id in interest_embeddings:
                filtered_interest_embeddings[paper.id] = interest_embeddings[paper.id]
        
        if filtered_interest_embeddings:
            logger.info(f"Generating recommendations for {source_type} using {len(filtered_interest_embeddings)} papers")
            return recommend_papers(
                papers,
                candidate_embeddings,
                filtered_interest_embeddings,
                interest_papers={paper.id: paper for paper in interest_papers[source_type]},
                top_k=top_k,
                seen_ids=seen_ids,
                use_seen_papers=use_seen_papers
            )
    
    logger.warning(f"No interest papers or embeddings found for type: {source_type}")
    return []

def get_recommendations(
    config: Dict[str, Any],
    papers: List[Paper] = None,
    interest_papers: Dict[str, List[Paper]] = None,
    use_seen_papers: bool = False
) -> Dict[str, List[Tuple[Paper, float, Paper]]]:
    """
    Main function to get paper recommendations based on config.
    
    Args:
        config: Configuration dictionary
        papers: List of candidate papers (if None, will load from saved embeddings)
        interest_papers: Dictionary of interest papers by type
        
    Returns:
        Dict[str, List[Tuple[Paper, float]]]: Dictionary mapping recommendation types to lists of (paper, score) tuples
    """
    # Load embeddings
    candidate_embeddings = load_embeddings('data/candidate_embeddings.npz')
    interest_embeddings = load_embeddings('data/interest_embeddings.npz')
    
    if not candidate_embeddings or not interest_embeddings:
        logger.error("Failed to load embeddings, cannot generate recommendations")
        return {}
    
    # Get top-k from config
    top_k = config.get('top_k', 10)
    
    # Load seen papers to avoid recommending the same papers again
    seen_ids = load_seen_papers(use_seen_papers=use_seen_papers)
    
    # If papers not provided, load them from the embeddings
    if papers is None:
        from app.fetcher import save_candidate_papers
        from app.reader import load_interest_papers
        
        # We need to load papers to get their metadata
        papers = []
        # Try to load from the most recent candidate papers file
        data_dir = 'data'
        candidate_files = [f for f in os.listdir(data_dir) if f.startswith('candidate_papers_') and f.endswith('.jsonl')]
        if candidate_files:
            # Get the most recent file
            latest_file = sorted(candidate_files)[-1]
            with open(os.path.join(data_dir, latest_file), 'r') as f:
                for line in f:
                    if line.strip():
                        paper_dict = json.loads(line)
                        papers.append(Paper.from_dict(paper_dict))
    
    # If interest papers not provided, load them
    if interest_papers is None:
        from app.reader import load_interest_papers
        interest_papers = load_interest_papers(config)
    
    # Generate recommendations for each type of interest papers
    recommendations = {}
    all_seen_ids = seen_ids.copy()
    
    # Recommendations based on my publications
    if 'my_pubs' in interest_papers and interest_papers['my_pubs']:
        my_pub_embeddings = {paper.id: interest_embeddings[paper.id] for paper in interest_papers['my_pubs'] if paper.id in interest_embeddings}
        my_pub_papers = {paper.id: paper for paper in interest_papers['my_pubs']}
        
        if my_pub_embeddings:
            logger.info(f"Getting recommendations based on {len(my_pub_embeddings)} publications")
            my_pub_recs = recommend_papers(
                papers,
                candidate_embeddings,
                my_pub_embeddings,
                interest_papers=my_pub_papers,
                top_k=top_k,
                seen_ids=all_seen_ids,
                use_seen_papers=use_seen_papers
            )
            recommendations['my_pubs'] = my_pub_recs
            
            # Update seen papers
            for paper, _, _ in my_pub_recs:
                all_seen_ids.add(paper.id)
    
    # Find all interesting_directions keys (they start with 'interesting_directions.')
    direction_keys = [key for key in interest_papers.keys() if key.startswith('zotero_collections.')]
    
    # If we have specific direction keys, use them
    if direction_keys:
        # Each direction gets the full top_k recommendations
        
        # Generate recommendations for each direction
        for direction_key in direction_keys:
            recommendations[direction_key] = get_recommendations_by_type(
                config, papers, interest_papers, candidate_embeddings, interest_embeddings,
                direction_key, top_k, all_seen_ids.copy(), use_seen_papers
            )
    # Fallback to legacy 'interesting_directions' if it exists
    elif 'interesting_directions' in interest_papers and interest_papers['interesting_directions']:
        interesting_embeddings = {paper.id: interest_embeddings[paper.id] for paper in interest_papers['interesting_directions'] if paper.id in interest_embeddings}
        interesting_papers_dict = {paper.id: paper for paper in interest_papers['interesting_directions']}
        
        if interesting_embeddings:
            logger.info(f"Getting recommendations based on {len(interesting_embeddings)} interesting papers")
            interesting_recs = recommend_papers(papers, candidate_embeddings, interesting_embeddings, interest_papers=interesting_papers_dict, top_k=top_k, seen_ids=all_seen_ids, use_seen_papers=use_seen_papers)
            recommendations['interesting_directions'] = interesting_recs
            
            # Update seen papers
            for paper, _, _ in interesting_recs:
                all_seen_ids.add(paper.id)
    
    # Update seen papers with all recommendations if the feature is enabled
    if use_seen_papers:
        # Make sure we collect all paper IDs from all recommendation lists
        for rec_list in recommendations.values():
            for paper, _, _ in rec_list:
                all_seen_ids.add(paper.id)
        save_seen_papers(all_seen_ids, use_seen_papers=use_seen_papers)
    return recommendations

if __name__ == "__main__":
    # For testing purposes
    import yaml
    from app.fetcher import fetch_papers
    from app.reader import load_interest_papers
    from app.embedder import embed_papers
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Fetch papers
    papers = fetch_papers(config)
    
    # Load interest papers
    interest_papers = load_interest_papers(config)
    
    # Embed papers
    candidate_embeddings = embed_papers(papers, config)
    
    # Generate recommendations
    recommendations = get_recommendations(config, papers, interest_papers)
    
    # Print recommendations
    print("\n===== Recommendations Based on My Publications =====\n")
    for i, (paper, score, ref_paper) in enumerate(recommendations.get('my_pubs', []), 1):
        print(f"{i}. {paper.title} (Score: {score:.4f})")
        print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print(f"   URL: {paper.url}")
        if ref_paper:
            print(f"   Similar to: {ref_paper.title}")
        print()
    
    # Print recommendations for each research direction
    for key, recs in recommendations.items():
        if key.startswith('interesting_directions.'):
            direction_name = key.split('.', 1)[1]  # Extract name after the dot
            print(f"\n===== Recommendations Based on {direction_name.replace('_', ' ').title()} =====\n")
            for i, (paper, score, ref_paper) in enumerate(recs, 1):
                print(f"{i}. {paper.title} (Score: {score:.4f})")
                print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
                print(f"   URL: {paper.url}")
                if ref_paper:
                    print(f"   Similar to: {ref_paper.title}")
                print()
    
    # Print legacy interesting directions if they exist
    if 'interesting_directions' in recommendations:
        print("\n===== Recommendations Based on Interesting Directions =====\n")
        for i, (paper, score, ref_paper) in enumerate(recommendations['interesting_directions'], 1):
            print(f"{i}. {paper.title} (Score: {score:.4f})")
            print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
            print(f"   URL: {paper.url}")
            if ref_paper:
                print(f"   Similar to: {ref_paper.title}")
            print()
