import os


class EmailConfig:
    EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
    EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
    EMAIL_USERNAME = os.getenv("EMAIL_USERNAME", "dogukangundogan5@gmail.com")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
    EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "Emotion Analysis API")
    EMAIL_FROM_ADDRESS = os.getenv("EMAIL_FROM_ADDRESS", "dogukangundogan5@gmail.com")
    EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS", "true").lower() == "true"
