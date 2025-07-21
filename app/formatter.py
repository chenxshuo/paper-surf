#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
formatter.py
Module for formatting paper recommendations into HTML digest.
"""

import os
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any
from app.models import Paper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('papersurf.formatter')

def format_paper_html(paper: Paper, score: float, reference_paper: Paper = None) -> str:
    """
    Format a single paper recommendation as HTML.
    
    Args:
        paper: The recommended paper
        score: Similarity score
        reference_paper: The paper this recommendation is based on (if available)
        
    Returns:
        str: HTML content for the paper
    """
    # Format authors (limit to 3 with ellipsis if more)
    if len(paper.authors) > 3:
        authors_str = ", ".join(paper.authors[:3]) + ", et al."
    else:
        authors_str = ", ".join(paper.authors)
    
    # Format abstract (truncate if too long)
    abstract = paper.abstract
    if len(abstract) > 800:
        abstract = abstract[:800] + "..."
    
    # Format score as percentage
    score_percent = int(score * 100)
    
    # Format paper details
    html = f"""            <div class="paper">
                <h3 class="paper-title">{paper.title}</h3>
                <div class="paper-meta">
                    <span class="paper-score">{score_percent}% Match</span>
                    <span class="paper-authors">{authors_str}</span>
                </div>"""
    
    # Add reference paper information if available
    if reference_paper:
        # Format reference paper authors
        if len(reference_paper.authors) > 3:
            ref_authors_str = ", ".join(reference_paper.authors[:3]) + ", et al."
        else:
            ref_authors_str = ", ".join(reference_paper.authors)
            
        html += f"""
                <div class="paper-reference">
                    <p class="reference-label">Recommended because it's similar to:</p>
                    <p class="reference-title"><a href="{reference_paper.url}" target="_blank">{reference_paper.title}</a></p>
                    <p class="reference-authors">{ref_authors_str}</p>
                </div>"""
    
    # Add abstract and link
    html += f"""
                <div class="paper-abstract">
                    {abstract}
                </div>
                <a href="{paper.url}" class="paper-link" target="_blank">Read Paper</a>
            </div>
"""
    
    return html

def format_html_digest(
    recommendations: Dict[str, List[Tuple[Paper, float, Paper]]],
    config: Dict[str, Any] = None,
    date_str: str = None,
    total_papers_examined: int = 0
) -> str:
    """
    Format paper recommendations into an HTML digest with separate sections for each type.
    
    Args:
        recommendations: Dictionary mapping recommendation types to lists of (paper, score, reference_paper) tuples
        config: Configuration dictionary
        date_str: Date string for the digest (defaults to today)
        
    Returns:
        str: HTML content of the digest
    """
    if not recommendations:
        logger.warning("No recommendations to format")
        return "<p>No paper recommendations available.</p>"
    
    # Get date string if not provided
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    # Get configuration values or use defaults
    if config is None:
        config = {}
    
    title = config.get('digest', {}).get('title', 'PaperSurf Daily Digest')
    subtitle = config.get('digest', {}).get('subtitle', 'Academic Paper Recommendations')
    
    # Start building HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - {date_str}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }}
        .header h1 {{
            margin-bottom: 5px;
            color: #2c3e50;
        }}
        .header p {{
            color: #7f8c8d;
            font-size: 1.1em;
        }}
        .section-header {{
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
            color: #2c3e50;
        }}
        .subsection-header {{
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #3498db;
            color: #2c3e50;
            font-size: 1.4em;
        }}
        .paper {{
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }}
        .paper-title {{
            font-size: 1.3em;
            margin-bottom: 10px;
            color: #2980b9;
        }}
        .paper-meta {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 10px;
        }}
        .paper-score {{
            display: inline-block;
            background-color: #2ecc71;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            margin-right: 10px;
        }}
        .paper-abstract {{
            margin-top: 10px;
            font-size: 0.95em;
            color: #555;
        }}
        .paper-link {{
            display: inline-block;
            margin-top: 10px;
            color: #3498db;
            text-decoration: none;
        }}
        .paper-link:hover {{
            text-decoration: underline;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            font-size: 0.9em;
            color: #7f8c8d;
        }}
        .no-recommendations {{
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
            text-align: center;
            color: #6c757d;
        }}
        .summary-section {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .summary-title {{
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #2c3e50;
        }}
        .summary-text {{
            line-height: 1.6;
            color: #555;
            font-size: 1.05em;
        }}
        .summary-text strong {{
            color: #2980b9;
            font-weight: bold;
        }}
        .toc {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .toc-title {{
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #2c3e50;
        }}
        .toc-list {{
            list-style-type: none;
            padding-left: 0;
        }}
        .toc-list li {{
            margin-bottom: 10px;
        }}
        .toc-list a {{
            color: #3498db;
            text-decoration: none;
        }}
        .toc-list a:hover {{
            text-decoration: underline;
        }}
        .paper-reference {{
            background-color: #f8f9fa;
            padding: 10px;
            border-left: 3px solid #3498db;
            margin: 10px 0;
            border-radius: 3px;
        }}
        .reference-label {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .reference-title {{
            font-weight: bold;
            margin-bottom: 3px;
        }}
        .reference-authors {{
            font-size: 0.9em;
            color: #555;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>{subtitle} - {date_str}</p>
    </div>
"""
    
    # Calculate summary statistics
    total_papers = sum(len(recs) for recs in recommendations.values())
    total_sections = len(recommendations)
    
    # Create table of contents
    html += """    <div class="toc">
        <h2 class="toc-title">Table of Contents</h2>
        <ul class="toc-list">
            <li><a href="#summary">Summary</a></li>
"""
    
    # Add TOC entries for each section
    # Always include My Publications section in TOC
    html += '            <li><a href="#my-publications">Based on My Publications</a></li>\n'
    
    # Add TOC entries for research directions
    direction_keys = [key for key in recommendations.keys() if key.startswith('zotero_collections.')]
    for key in direction_keys:
        direction_name = key.split('.', 1)[1]  # Extract name after the dot
        if recommendations[key]:
            section_id = direction_name.replace('_', '-').lower()
            section_title = direction_name.replace('_', ' ').title()
            html += f'            <li><a href="#{section_id}">Based on {section_title}</a></li>\n'
    
    # Add TOC entry for legacy interesting directions
    if 'zotero_collections' in recommendations and recommendations['zotero_collections']:
        html += '            <li><a href="#zotero-collections">Based on Zotero Collections (Legacy)</a></li>\n'
    
    html += """        </ul>
    </div>
"""
    
    # Add summary section
    html += """    <div class="summary-section" id="summary">
        <h2 class="summary-title">Summary</h2>
"""
    
    # Create section names dictionary for the summary paragraph
    section_names = {}
    for key, papers_list in recommendations.items():
        if key == 'my_pubs':
            section_names[key] = "Based on My Publications"
        elif key.startswith('zotero_collections.'):
            direction_name = key.split('.', 1)[1]
            section_names[key] = f"Based on {direction_name.replace('_', ' ').title()}"
        elif key == 'zotero_collections':
            section_names[key] = "Based on Zotero Collections (Legacy)"
        else:
            section_names[key] = key.replace('_', ' ').title()
    
    # Create section details list
    section_details = []
    for key, papers_list in recommendations.items():
        section_details.append(f"<strong>{len(papers_list)}</strong> papers in {section_names[key]}")
    
    # Format section details with proper grammar
    section_details_text = ""
    if section_details:
        if len(section_details) == 1:
            section_details_text = section_details[0]
        elif len(section_details) == 2:
            section_details_text = f"{section_details[0]} and {section_details[1]}"
        else:
            section_details_text = ", ".join(section_details[:-1]) + f", and {section_details[-1]}"
    
    # Build the complete summary paragraph
    summary_html = f"""        <p class="summary-text">
            This digest was generated on <strong>{date_str}</strong> after examining <strong>{total_papers_examined}</strong> papers from arXiv. 
            A total of <strong>{total_papers}</strong> papers were recommended across <strong>{total_sections}</strong> categories: 
            {section_details_text}.
        </p>
"""
    
    html += summary_html + "    </div>\n"
    
    # Add section for my publications
    if 'my_pubs' in recommendations and recommendations['my_pubs']:
        html += """    <div class="recommendation-section">
        <h2 class="section-header" id="my-publications">Based on Your Publications</h2>
        <p class="section-description">Papers similar to your published work</p>
"""
        
        # Add each paper recommendation for this section
        for paper, score, ref_paper in recommendations['my_pubs']:
            html += format_paper_html(paper, score, ref_paper)
        
        html += "    </div>\n"
    else:
        html += """    <div class="recommendation-section">
        <h2 class="section-header" id="my-publications">Based on Your Publications</h2>
        <p class="section-description">Papers similar to your published work</p>
        <div class="no-recommendations">
            <p>No recommendations available for this category.</p>
        </div>
    </div>
"""
    
    # Start the research interests section
    html += """    <div class="recommendation-section">
        <h2 class="section-header">Based on Your Research Interests</h2>
        <p class="section-description">Papers aligned with your current research directions</p>
"""
    
    # Find all interesting_directions keys (they start with 'interesting_directions.')
    direction_keys = [key for key in recommendations.keys() if key.startswith('zotero_collections.')]
    
    if direction_keys:
        # Add subsections for each research direction
        for direction_key in direction_keys:
            direction_name = direction_key.split('.', 1)[1]  # Extract name after the dot
            display_name = direction_name.replace('_', ' ').title()
            section_id = direction_name.replace('_', '-').lower()
            
            html += f"""        <h3 class="section-header" id="{section_id}">Based on {display_name}</h3>
        <div class="recommendation-subsection">
"""
            
            # Add each paper recommendation for this direction
            for paper, score, ref_paper in recommendations[direction_key]:
                html += format_paper_html(paper, score, ref_paper)
            
            html += "        </div>\n"
    
    # Handle legacy 'zotero_collections' if it exists
    elif 'zotero_collections' in recommendations and recommendations['zotero_collections']:
        html += """        <div class="recommendation-subsection" id="zotero-collections">
            <h3 class="subsection-header">General Research Interests</h3>
"""
        
        # Add each paper recommendation for this section
        for paper, score, ref_paper in recommendations['interesting_directions']:
            html += format_paper_html(paper, score, ref_paper)
        
        html += "        </div>\n"
    else:
        html += """        <div class="no-recommendations">
            <p>No recommendations available for research interests.</p>
        </div>
"""
    
    html += "    </div>\n"
    
    # Add footer
    html += f"""    <div class="footer">
        <p>Generated by PaperSurf on {date_str}</p>
    </div>
</body>
</html>
"""
    
    return html

def save_html_digest(
    html_content: str,
    output_dir: str = 'output',
    filename: str = None
) -> str:
    """
    Save HTML digest to a file.
    
    Args:
        html_content: HTML content to save
        output_dir: Directory to save the file
        filename: Filename to use (defaults to digest_YYYY-MM-DD.html)
        
    Returns:
        str: Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if not filename:
        date_str = datetime.now().strftime('%Y-%m-%d')
        filename = f"digest_{date_str}.html"
    
    # Full path to save the file
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML digest saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving HTML digest: {e}")
        return None

if __name__ == "__main__":
    # For testing purposes
    import yaml
    from app.recommender import get_recommendations
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate recommendations
    recommendations = get_recommendations(config)
    
    # Format HTML digest
    html_content = format_html_digest(recommendations, config)
    
    # Save HTML digest
    filepath = save_html_digest(html_content)
    
    # Print path to saved file
    if filepath:
        print(f"HTML digest saved to {filepath}")
    else:
        print("Failed to save HTML digest")
