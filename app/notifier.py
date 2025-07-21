#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
notifier.py
Module for sending notifications (email) with paper recommendations.
"""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('papersurf.notifier')

def send_email_notification(
    html_content: str,
    config: Dict[str, Any],
    subject: Optional[str] = None,
    date_str: Optional[str] = None,
    attach_html: bool = True
) -> bool:
    """
    Send an email notification with the HTML digest.
    
    Args:
        html_content: HTML content to send
        config: Configuration dictionary with email settings
        subject: Email subject (optional)
        date_str: Date string for the subject (optional)
        attach_html: Whether to attach the HTML content as a file (default: True)
        
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    # Check if email configuration exists
    if 'notification' not in config or config['notification'].get('method') != 'email':
        logger.error("Email notification not configured")
        return False
    
    email_config = config['notification'].get('email', {})
    
    # Check required email configuration
    required_fields = ['smtp_server', 'smtp_port', 'sender', 'receiver']
    for field in required_fields:
        if field not in email_config:
            logger.error(f"Missing required email configuration: {field}")
            return False
    
    # Extract email configuration
    smtp_server = email_config['smtp_server']
    smtp_port = email_config['smtp_port']
    sender_email = os.environ.get('EMAIL_SENDER')
    # Get password from environment variable instead of config
    password = os.environ.get('EMAIL_PASSWORD')
    if not password:
        logger.error("EMAIL_PASSWORD environment variable not set in .env file")
        return False
    receiver_email = os.environ.get('EMAIL_RECEIVER')
    
    # Create message
    msg = MIMEMultipart('alternative')
    
    # Set subject
    if not subject:
        digest_title = config.get('digest', {}).get('title', 'PaperSurf Daily Digest')
        if date_str:
            subject = f"{digest_title} - {date_str}"
        else:
            subject = f"{digest_title} - {datetime.now().strftime('%Y-%m-%d')}"
            
    # Set date string for filename if not provided
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    
    # Attach HTML content as viewable email body
    html_part = MIMEText(html_content, 'html')
    msg.attach(html_part)
    
    # Attach HTML content as a file if requested
    if attach_html:
        # Create filename with date
        filename = f"papersurf_digest_{date_str}.html"
        
        # Create attachment
        attachment = MIMEApplication(html_content.encode('utf-8'), _subtype='html')
        attachment.add_header('Content-Disposition', 'attachment', filename=filename)
        msg.attach(attachment)
        
        logger.info(f"Attached HTML digest as {filename}")
    
    # Send email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        
        logger.info(f"Email notification sent to {receiver_email}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")
        return False

def send_notification(
    html_content: str,
    config: Dict[str, Any],
    date_str: Optional[str] = None,
    attach_html: bool = True
) -> bool:
    """
    Send a notification with the HTML digest using the configured method.
    
    Args:
        html_content: HTML content to send
        config: Configuration dictionary
        date_str: Date string for the notification (optional)
        attach_html: Whether to attach the HTML content as a file (default: True)
        
    Returns:
        bool: True if notification was sent successfully, False otherwise
    """
    # Check notification configuration
    if 'notification' not in config:
        logger.error("Notification not configured")
        return False
    
    # Get notification method
    method = config['notification'].get('method', 'email')
    
    # Send notification based on method
    if method == 'email':
        return send_email_notification(html_content, config, date_str=date_str, attach_html=attach_html)
    else:
        logger.error(f"Unsupported notification method: {method}")
        return False

if __name__ == "__main__":
    # For testing purposes
    import yaml
    from app.formatter import format_html_digest
    from app.recommender import get_recommendations
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    html_content = "<p>Test email content</p>"

    # Send notification
    if send_notification(html_content, config):
        print("Notification sent successfully!")
    else:
        print("Failed to send notification")

    # # Get recommendations
    # recommendations = get_recommendations(config)
    #
    # if recommendations:
    #     # Format recommendations into HTML
    #     html_content = format_html_digest(recommendations, config)
    #
    #     # Send notification
    #     if send_notification(html_content, config):
    #         print("Notification sent successfully!")
    #     else:
    #         print("Failed to send notification")
    # else:
    #     print("No recommendations available for testing")
