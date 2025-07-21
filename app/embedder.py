#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
embedder.py
Module for embedding papers using the Specter2 model.
"""

import os
import torch
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union
from app.models import Paper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('papersurf.embedder')

class PaperEmbedder:
    """
    Class for embedding papers using the Specter2 model.
    """
    
    def __init__(self, model_name: str = "allenai/specter2_base", device: str = None):
        """
        Initialize the embedder with the specified model.
        
        Args:
            model_name: Name of the HuggingFace model to use
            device: Device to run the model on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        logger.info(f"Initializing PaperEmbedder with model {model_name} on {self.device}")
    
    def load_model(self):
        """
        Load the model and tokenizer.
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"Loading model {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def embed_paper(self, paper: Paper) -> Optional[np.ndarray]:
        """
        Embed a single paper.
        
        Args:
            paper: Paper object to embed
            
        Returns:
            np.ndarray: Embedding vector or None if embedding failed
        """
        if self.model is None or self.tokenizer is None:
            if not self.load_model():
                return None
        
        try:
            # Prepare input text (title + abstract)
            title = paper.title or ""
            abstract = paper.abstract or ""
            text = f"{title} {abstract}"
            
            # Tokenize and get embedding
            inputs = self.tokenizer(text, padding=True, truncation=True, 
                                   return_tensors="pt", max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding as paper embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings[0]  # Return the first (and only) embedding
        
        except Exception as e:
            logger.error(f"Error embedding paper {paper.id}: {e}")
            return None
    
    def embed_papers(self, papers: List[Paper], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Embed a list of papers.
        
        Args:
            papers: List of Paper objects to embed
            batch_size: Batch size for processing
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping paper IDs to embeddings
        """
        if self.model is None or self.tokenizer is None:
            if not self.load_model():
                return {}
        
        embeddings = {}
        total_papers = len(papers)
        
        try:
            # Process papers in batches
            for i in range(0, total_papers, batch_size):
                batch_papers = papers[i:i+batch_size]
                batch_ids = [p.id for p in batch_papers]
                
                # Prepare batch input
                titles = [p.title or "" for p in batch_papers]
                abstracts = [p.abstract or "" for p in batch_papers]
                texts = [f"{t} {a}" for t, a in zip(titles, abstracts)]
                
                # Tokenize
                inputs = self.tokenizer(texts, padding=True, truncation=True, 
                                       return_tensors="pt", max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Store embeddings
                for j, paper_id in enumerate(batch_ids):
                    embeddings[paper_id] = batch_embeddings[j]
                
                logger.info(f"Embedded batch {i//batch_size + 1}/{(total_papers-1)//batch_size + 1} "
                           f"({len(batch_papers)} papers)")
        
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
        
        logger.info(f"Successfully embedded {len(embeddings)}/{total_papers} papers")
        return embeddings

def embed_papers(
    papers: Union[List[Paper], Dict[str, List[Paper]]],
    config: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Embed papers using the model specified in config.
    
    Args:
        papers: List of Paper objects or dictionary mapping interest types to lists of Paper objects
        config: Configuration dictionary
        
    Returns:
        Dict[str, np.ndarray]: Dictionary mapping paper IDs to embeddings
    """
    model_name = config.get('embedding_model', 'allenai/specter2_base')
    
    # Initialize embedder
    embedder = PaperEmbedder(model_name=model_name)
    
    # Handle both list and dictionary input formats
    if isinstance(papers, dict):
        # Flatten the dictionary of papers into a single list
        all_papers = []
        for paper_type, paper_list in papers.items():
            logger.info(f"Adding {len(paper_list)} papers from category '{paper_type}'")
            all_papers.extend(paper_list)
        
        logger.info(f"Embedding {len(all_papers)} papers from all interest categories")
        return embedder.embed_papers(all_papers)
    else:
        # Input is already a list of papers
        return embedder.embed_papers(papers)

def save_embeddings(embeddings: Dict[str, np.ndarray], output_path: str = 'data/embeddings.npz'):
    """
    Save embeddings to a file.
    
    Args:
        embeddings: Dictionary mapping paper IDs to embeddings
        output_path: Path to save the embeddings
        
    Returns:
        str: Path to the saved file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to format suitable for numpy save
    ids = list(embeddings.keys())
    vectors = np.array([embeddings[id_] for id_ in ids])
    
    # Save embeddings
    np.savez(output_path, ids=ids, vectors=vectors)
    logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")
    
    return output_path

def load_embeddings(input_path: str = 'data/embeddings.npz') -> Dict[str, np.ndarray]:
    """
    Load embeddings from a file.
    
    Args:
        input_path: Path to load the embeddings from
        
    Returns:
        Dict[str, np.ndarray]: Dictionary mapping paper IDs to embeddings
    """
    if not os.path.exists(input_path):
        logger.warning(f"Embeddings file not found: {input_path}")
        return {}
    
    try:
        data = np.load(input_path, allow_pickle=True)
        ids = data['ids']
        vectors = data['vectors']
        
        embeddings = {id_: vector for id_, vector in zip(ids, vectors)}
        logger.info(f"Loaded {len(embeddings)} embeddings from {input_path}")
        
        return embeddings
    
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        return {}

if __name__ == "__main__":
    # For testing purposes
    import yaml
    from app.reader import load_interest_papers
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load interest papers
    interest_papers = load_interest_papers(config)
    
    # Print interest paper categories
    print("Interest paper categories:")
    for category, papers in interest_papers.items():
        print(f"  - {category}: {len(papers)} papers")
    
    # Embed interest papers
    embeddings = embed_papers(interest_papers, config)
    
    # Save embeddings
    if embeddings:
        output_path = save_embeddings(embeddings, 'data/interest_embeddings.npz')
        print(f"Saved {len(embeddings)} embeddings to {output_path}")
    else:
        print("No embeddings generated")
