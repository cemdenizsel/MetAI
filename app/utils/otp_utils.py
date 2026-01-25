"""
OTP (One-Time Password) Utilities

Handles OTP generation, validation, and management.
"""

import random
import string
import secrets
from typing import Tuple


def generate_numeric_otp(length: int = 6) -> str:
    """
    Generate a numeric OTP code.
    
    Args:
        length: Length of the OTP code (default: 6)
        
    Returns:
        Numeric OTP code as string
    """
    return ''.join(random.choices(string.digits, k=length))


def generate_alphanumeric_otp(length: int = 8) -> str:
    """
    Generate an alphanumeric OTP code.
    
    Args:
        length: Length of the OTP code (default: 8)
        
    Returns:
        Alphanumeric OTP code as string
    """
    characters = string.ascii_uppercase + string.digits
    return ''.join(random.choices(characters, k=length))


def generate_secure_otp(length: int = 6) -> str:
    """
    Generate a cryptographically secure numeric OTP code.
    
    Args:
        length: Length of the OTP code (default: 6)
        
    Returns:
        Secure numeric OTP code as string
    """
    # Use secrets module for cryptographically secure random generation
    return ''.join(secrets.choice(string.digits) for _ in range(length))


def validate_otp_format(otp: str, expected_length: int = 6) -> bool:
    """
    Validate OTP format.
    
    Args:
        otp: OTP code to validate
        expected_length: Expected length of the OTP
        
    Returns:
        True if format is valid, False otherwise
    """
    if not otp:
        return False
    
    # Check if OTP is numeric and has correct length
    return otp.isdigit() and len(otp) == expected_length


def generate_otp_with_metadata(length: int = 6) -> Tuple[str, dict]:
    """
    Generate OTP with metadata for logging/debugging.
    
    Args:
        length: Length of the OTP code
        
    Returns:
        Tuple of (otp_code, metadata_dict)
    """
    import time
    
    otp_code = generate_secure_otp(length)
    metadata = {
        "generated_at": time.time(),
        "length": length,
        "type": "numeric",
        "secure": True
    }
    
    return otp_code, metadata


class OTPGenerator:
    """OTP Generator class with configurable settings."""
    
    def __init__(self, length: int = 6, numeric_only: bool = True, secure: bool = True):
        self.length = length
        self.numeric_only = numeric_only
        self.secure = secure
    
    def generate(self) -> str:
        """Generate OTP based on configuration."""
        if self.secure:
            if self.numeric_only:
                return generate_secure_otp(self.length)
            else:
                return generate_alphanumeric_otp(self.length)
        else:
            if self.numeric_only:
                return generate_numeric_otp(self.length)
            else:
                return generate_alphanumeric_otp(self.length)
    
    def validate_format(self, otp: str) -> bool:
        """Validate OTP format based on configuration."""
        if not otp or len(otp) != self.length:
            return False
        
        if self.numeric_only:
            return otp.isdigit()
        else:
            return otp.isalnum() and otp.isupper()


# Default OTP generator for password reset
password_reset_otp_generator = OTPGenerator(length=6, numeric_only=True, secure=True)


def generate_password_reset_otp() -> str:
    """Generate OTP specifically for password reset."""
    return password_reset_otp_generator.generate()


def validate_password_reset_otp_format(otp: str) -> bool:
    """Validate password reset OTP format."""
    return password_reset_otp_generator.validate_format(otp)