"""
Spec 10: Email notification system with templates
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.config import settings
from app.utils.retry import retry_with_backoff
import logging

logger = logging.getLogger(__name__)


@retry_with_backoff(max_attempts=3, initial_delay=2.0)
def send_email(to_email: str, subject: str, body: str):
    """
    Send email with retry logic
    
    Spec 10: Trade executed email with retry on failure
    """
    if not settings.SMTP_USER or not settings.SMTP_PASS:
        logger.warning("SMTP credentials not configured, skipping email send")
        return
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = settings.SMTP_USER
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Add HTML and plain text parts
        part = MIMEText(body, 'plain')
        msg.attach(part)
        
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASS)
            server.send_message(msg)
        
        logger.info(f"Email sent successfully to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        raise


def send_trade_execution_email(trade_data: dict):
    """
    Spec 10: Trade executed email template
    """
    subject = f"JarvisTrade — Trade Executed: {trade_data['symbol']} ({trade_data['mode']})"
    
    body = f"""
Trade ID: {trade_data['trade_id']}
User: {trade_data['email']}
Mode: {trade_data['mode']}
Symbol: {trade_data['symbol']}
Entry: {trade_data['entry_price']} @ {trade_data['entry_ts']} UTC
Qty: {trade_data['qty']}
Stop: {trade_data['stop']}
Target: {trade_data['target']}
Probability: {trade_data['prob']:.2f}
Model: {trade_data['model_name']}
Features: {trade_data.get('top_features_json', 'N/A')}
PnL: {trade_data.get('pnl', 'Open')}
Notes: {trade_data['reason']}

---
This is an automated notification from JarvisTrade.
"""
    
    send_email(trade_data['email'], subject, body)
