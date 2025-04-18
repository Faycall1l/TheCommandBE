import re

def clean_media_plan_text(text):
    """
    Cleans marketing/media plan text by:
    - Removing all * and # characters
    - Maintaining proper indentation
    - Keeping consistent structure
    - Preserving email subject formatting
    """
    # Remove all * and # characters from headings
    text = re.sub(r'^[#*]+\s*(.*?Week\s*\d+.*?)\n', lambda m: f"{m.group(1).strip('#* ')}\n\n", text, flags=re.IGNORECASE|re.MULTILINE)
    
    # Remove stars from section headers (Social Media, Promo, etc.)
    text = re.sub(r'^[#*]+([A-Za-z\s]+:)[#*]*', lambda m: f"{m.group(1).strip('#* ')}", text, flags=re.MULTILINE)
    
    # Standardize bullet points with proper indentation (using - instead of *)
    text = re.sub(r'^(\s*)[\*\-](\s+)', lambda m: '  ' * (len(m.group(1))//2) + '- ', text, flags=re.MULTILINE)
    
    # Clean email subjects (keep "Email Subject:" formatting)
    def clean_email_subject(match):
        subject = match.group(3).strip('"*\'')
        return f"Email Subject: {subject}\n"
    
    text = re.sub(
        r'(Email:\s*Subject:\s*|Email Subject:\s*)(["\*]?)(.*?)(["\*]?\n)',
        clean_email_subject,
        text,
        flags=re.IGNORECASE
    )
    
    # Remove all remaining standalone stars
    text = re.sub(r'\*', '', text)
    
    # Remove all remaining standalone hash symbols
    text = re.sub(r'#', '', text)
    
    # Fix excessive whitespace
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double
    
    # Clean promo codes
    text = re.sub(r'"(code\s*[A-Z0-9]+)"', r'\1', text)
    
    # Fix date formats
    text = re.sub(r'(\w{3,9})\s(\d{1,2})(st|nd|rd|th)', r'\1 \2', text)
    
    return text.strip()