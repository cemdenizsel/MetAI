import smtplib
from pathlib import Path
from typing import Optional

from email.message import EmailMessage
from email.headerregistry import Address
from enum import Enum
from dotenv import load_dotenv

from config.email_config import EmailConfig

load_dotenv()


class EmailType(Enum):
    General = "/Users/dogukangundogan/Desktop/Dev/random-feature-representation-boosting/app/api/templates/general.html"
    Result = "/Users/dogukangundogan/Desktop/Dev/random-feature-representation-boosting/app/api/templates/meeting.html"


def read_email_template(email_type: EmailType) -> Optional[str]:
    """
    Read email template from file path specified in EmailType enum.
    
    Args:
        email_type: EmailType enum containing the full path to template file
    
    Returns:
        Template content as string or None if file not found
    """
    try:
        template_path = Path(email_type.value)
        
        if template_path.exists():
            return template_path.read_text(encoding='utf-8')
                
        return None
    except Exception as e:
        print(f"Error reading email template {email_type.name}: {e}")
        return None


def create_default_template(content: str, template_type: str = "html") -> str:
    """Create a default email template wrapper."""
    if template_type == "html":
        return f"""
        <html>
            <head>
                <meta charset="utf-8">
                <title>{EmailConfig.EMAIL_FROM_NAME}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; text-align: center; border-radius: 5px 5px 0 0; }}
                    .content {{ background-color: #ffffff; padding: 20px; border-radius: 0 0 5px 5px; }}
                    .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h2>{EmailConfig.EMAIL_FROM_NAME}</h2>
                    </div>
                    <div class="content">
                        {content}
                    </div>
                    <div class="footer">
                        <p>This email was sent automatically. Please do not reply.</p>
                    </div>
                </div>
            </body>
        </html>
        """
    else:
        return f"""
{EmailConfig.EMAIL_FROM_NAME}
{'=' * len(EmailConfig.EMAIL_FROM_NAME)}

{content}

---
This email was sent automatically. Please do not reply.
        """

def send_email(
    to: str, 
    subject: str, 
    body: str, 
    email_type: EmailType = EmailType.General,
    template_variables: Optional[dict] = None
) -> bool:
    """
    Send email with optional template support.
    
    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body content
        email_type: Type of email for template organization
        template_variables: Variables to substitute in template
        
    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        # Create the message
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = Address(EmailConfig.EMAIL_FROM_NAME, EmailConfig.EMAIL_FROM_ADDRESS.split('@')[0], EmailConfig.EMAIL_FROM_ADDRESS.split('@')[1])
        msg['To'] = to
        
        # Prepare template variables
        if template_variables is None:
            template_variables = {}
        template_variables.update({
            'to': to,
            'subject': subject,
            'body': body
        })
        
        # Try to load template from EmailType enum
        template_content = read_email_template(email_type)
        if template_content:
            try:
                # Handle double-brace template format {{VARIABLE}}
                formatted_content = template_content
                for key, value in template_variables.items():
                    formatted_content = formatted_content.replace(f"{{{{{key}}}}}", str(value))
                
                # Determine if template is HTML
                if template_content.strip().startswith('<'):
                    msg.set_content(body)  # Plain text fallback first
                    msg.add_alternative(formatted_content, subtype='html')
                else:
                    msg.set_content(formatted_content)
            except KeyError as e:
                print(f"Missing template variable: {e}")
                # Fall back to default template
                html_content = create_default_template(body, "html")
                msg.set_content(body)
                msg.add_alternative(html_content, subtype='html')
        else:
            # Template not found, use default wrapper
            html_content = create_default_template(body, "html")
            msg.set_content(body)
            msg.add_alternative(html_content, subtype='html')
        
        # Send the message
        if EmailConfig.EMAIL_HOST == "localhost" and not EmailConfig.EMAIL_USERNAME:
            # Local SMTP server (development)
            with smtplib.SMTP(EmailConfig.EMAIL_HOST, EmailConfig.EMAIL_PORT) as server:
                server.send_message(msg)
        else:
            # External SMTP server (production)
            with smtplib.SMTP(EmailConfig.EMAIL_HOST, EmailConfig.EMAIL_PORT) as server:
                if EmailConfig.EMAIL_USE_TLS:
                    server.starttls()
                if EmailConfig.EMAIL_USERNAME and EmailConfig.EMAIL_PASSWORD:
                    server.login(EmailConfig.EMAIL_USERNAME, EmailConfig.EMAIL_PASSWORD)
                server.send_message(msg)
        
        print(f"Email sent successfully to {to}")
        return True
        
    except Exception as e:
        print(f"Failed to send email to {to}: {e}")
        return False


def send_password_reset_email(to: str, reset_token: str, reset_url: str) -> bool:
    """Send password reset email."""
    subject = "Password Reset Request"
    template_variables = {
        'EMAIL_TITLE': 'Password Reset Request',
        'RECIPIENT_NAME': to.split('@')[0],
        'MESSAGE_CONTENT': f"""You have requested a password reset for your account.<br><br>
        Your password reset code is: <strong>{reset_token}</strong><br><br>
        Please use this code to reset your password.<br>
        The code will expire in 10 minutes.<br><br>
        If you did not request this reset, please ignore this email.""",
        'CALL_TO_ACTION_TEXT': 'Reset Password',
        'CALL_TO_ACTION_URL': reset_url,
        'CTA_DISPLAY': '' if reset_url and reset_url != '#' else 'display: none;'
    }
    
    # Default body if template is not found
    body = f"""
    You have requested a password reset for your account.
    
    Your password reset code is: {reset_token}
    
    Please use this code to reset your password. 
    The code will expire in 10 minutes.
    
    If you did not request this reset, please ignore this email.
    """
    
    return send_email(
        to=to,
        subject=subject,
        body=body,
        email_type=EmailType.General,
        template_variables=template_variables
    )


def send_welcome_email(to: str, username: str) -> bool:
    """Send welcome email to new users."""
    subject = "Welcome to Emotion Analysis API"
    template_variables = {
        'EMAIL_TITLE': 'Welcome to NexCoach',
        'RECIPIENT_NAME': username,
        'MESSAGE_CONTENT': f"""Welcome to Emotion Analysis API, {username}!<br><br>
        Your account has been successfully created with a Basic package.<br>
        You can now start using our emotion recognition services.<br><br>
        Thank you for joining us!""",
        'CALL_TO_ACTION_TEXT': '',
        'CALL_TO_ACTION_URL': '',
        'CTA_DISPLAY': 'display: none;'
    }
    
    # Default body if template is not found
    body = f"""
    Welcome to Emotion Analysis API, {username}!
    
    Your account has been successfully created with a Basic package.
    You can now start using our emotion recognition services.
    
    Thank you for joining us!
    """
    
    return send_email(
        to=to,
        subject=subject,
        body=body,
        email_type=EmailType.General,
        template_variables=template_variables
    )


def send_meeting_result_email(to: str, meeting_results: dict) -> bool:
    """Send meeting analysis results using Result template."""
    subject = "Meeting Analysis Results"
    meeting_id = meeting_results.get('meeting_id', 'Unknown')
    analysis_date = meeting_results.get('analysis_date', 'Unknown')
    
    template_variables = {
        'EMAIL_TITLE': 'Meeting Analysis Results',
        'RECIPIENT_NAME': to.split('@')[0],
        'MESSAGE_CONTENT': f"""Your meeting analysis results are ready!<br><br>
        <strong>Meeting ID:</strong> {meeting_id}<br>
        <strong>Analysis Date:</strong> {analysis_date}<br><br>
        Please find the detailed analysis results attached or available in your dashboard.""",
        'CALL_TO_ACTION_TEXT': 'View Results',
        'CALL_TO_ACTION_URL': '#',
        'CTA_DISPLAY': 'display: none;'
    }
    
    # Default body if template is not found
    body = f"""
    Your meeting analysis results are ready!
    
    Meeting ID: {meeting_id}
    Analysis Date: {analysis_date}
    
    Please find the detailed analysis results attached or available in your dashboard.
    """
    
    return send_email(
        to=to,
        subject=subject,
        body=body,
        email_type=EmailType.Result,
        template_variables=template_variables
    )
