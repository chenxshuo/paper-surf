#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
models.py
Core data models for PaperSurf.
"""

import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Union


class Paper:
    """
    Represents an academic paper with its metadata and content.
    """
    
    def __init__(
        self,
        id: str,
        title: str,
        abstract: str,
        authors: List[str],
        categories: List[str],
        published: str,
        updated: str = None,
        url: str = None,
        pdf_url: str = None,
        source: str = None,
        embedding: List[float] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a Paper object.
        
        Args:
            id: Unique identifier for the paper
            title: Title of the paper
            abstract: Abstract text
            authors: List of author names
            categories: List of subject categories
            published: Publication date (ISO format string)
            updated: Last updated date (ISO format string)
            url: URL to the paper's webpage
            pdf_url: URL to the paper's PDF
            source: Source of the paper (e.g., 'arxiv', 'user_interest')
            embedding: Vector embedding of the paper (if available)
            metadata: Additional metadata as key-value pairs
        """
        self.id = id
        self.title = self._clean_text(title)
        self.abstract = self._clean_text(abstract)
        self.authors = authors
        self.categories = categories
        self.published = published
        self.updated = updated or published
        self.url = url
        self.pdf_url = pdf_url
        self.source = source
        self.embedding = embedding
        self.metadata = metadata or {}
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text by removing extra whitespace and newlines."""
        if not text:
            return ""
        return " ".join(text.split())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Paper':
        """
        Create a Paper object from a dictionary.
        
        Args:
            data: Dictionary containing paper data
            
        Returns:
            Paper: A new Paper instance
        """
        return cls(
            id=data.get('id', ''),
            title=data.get('title', ''),
            abstract=data.get('abstract', ''),
            authors=data.get('authors', []),
            categories=data.get('categories', []),
            published=data.get('published', ''),
            updated=data.get('updated', None),
            url=data.get('url', None),
            pdf_url=data.get('pdf_url', None),
            source=data.get('source', None),
            embedding=data.get('embedding', None),
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def from_arxiv_result(cls, result: Any) -> 'Paper':
        """
        Create a Paper object from an arxiv.Result object.
        
        Args:
            result: An arxiv.Result object
            
        Returns:
            Paper: A new Paper instance
        """
        return cls(
            id=result.entry_id.split('/')[-1],
            title=result.title,
            abstract=result.summary,
            authors=[author.name for author in result.authors],
            categories=result.categories,
            published=result.published.isoformat(),
            updated=result.updated.isoformat(),
            url=result.entry_id,
            pdf_url=result.pdf_url,
            source='arxiv'
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Paper object to a dictionary.
        
        Returns:
            Dict: Dictionary representation of the paper
        """
        result = {
            'id': self.id,
            'title': self.title,
            'abstract': self.abstract,
            'authors': self.authors,
            'categories': self.categories,
            'published': self.published,
            'updated': self.updated,
            'source': self.source
        }
        
        # Add optional fields if they exist
        if self.url:
            result['url'] = self.url
        if self.pdf_url:
            result['pdf_url'] = self.pdf_url
        if self.embedding:
            result['embedding'] = self.embedding
        if self.metadata:
            result['metadata'] = self.metadata
            
        return result
    
    def to_json(self) -> str:
        """
        Convert the Paper object to a JSON string.
        
        Returns:
            str: JSON representation of the paper
        """
        return json.dumps(self.to_dict())
    
    def __str__(self) -> str:
        """String representation of the paper."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return f"{self.title} by {authors_str} ({self.id})"
    
    def __repr__(self) -> str:
        """Detailed representation of the paper."""
        return f"Paper(id='{self.id}', title='{self.title[:30]}...', source='{self.source}')"
