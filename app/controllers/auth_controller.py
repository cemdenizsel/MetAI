"""
Authentication Controller

Handles user authentication endpoints including registration, login,
password reset functionality.
"""

from datetime import timedelta
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import EmailStr

from models.user_models import UserRegistrationRequest, UserLoginRequest, TokenResponse, UserResponse, MessageResponse, PasswordResetRequest, ChangePasswordRequest
from utils.auth import create_access_token, get_current_user_email, ACCESS_TOKEN_EXPIRE_MINUTES, get_password_hash
from services.user_service import get_user_service, UserService
from services.redis_service import get_redis_service, RedisService
from utils.otp_utils import generate_password_reset_otp, validate_password_reset_otp_format
from utils.email import send_password_reset_email, send_welcome_email

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="Register a new user with basic package by default. Email must be unique."
)
async def register_user(
    user_data: UserRegistrationRequest,
    user_service: UserService = Depends(get_user_service)
):
    """
    Register a new user with the following features:
    - Automatic basic package assignment
    - Email uniqueness validation
    - Password hashing
    - JWT token generation
    """
    
    # Check if email already exists
    if await user_service.email_exists(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username already exists
    if await user_service.username_exists(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create user with basic package
    success = await user_service.create_user(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password,
        package="Basic"
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )
    
    # Get created user for response
    user_doc = await user_service.get_user_by_email(user_data.email)
    
    # Send welcome email
    try:
        send_welcome_email(user_data.email, user_data.username)
    except Exception as e:
        print(f"Failed to send welcome email: {e}")
        # Don't fail registration if email fails
    
    # Generate JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_data.email},
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Convert to seconds
        user=user_service.doc_to_api_response(user_doc)
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User login",
    description="Authenticate user and return JWT access token"
)
async def login_user(
    login_data: UserLoginRequest,
    user_service: UserService = Depends(get_user_service)
):
    """
    Authenticate user and return JWT token.
    Token contains user email for identification.
    """
    
    # Authenticate user
    user_doc = await user_service.authenticate_user(login_data.email, login_data.password)
    if not user_doc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Generate JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_doc["email"]},
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Convert to seconds
        user=user_service.doc_to_api_response(user_doc)
    )


@router.get(
    "/profile",
    response_model=UserResponse,
    summary="Get user profile",
    description="Get current user's profile information"
)
async def get_user_profile(
    current_user_email: str = Depends(get_current_user_email),
    user_service: UserService = Depends(get_user_service)
):
    """
    Get current user's profile information from JWT token.
    """
    
    # Get user by email from token
    user_doc = await user_service.get_user_by_email(current_user_email)
    if not user_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(**user_service.doc_to_api_response(user_doc))


@router.post(
    "/reset-password-request",
    response_model=MessageResponse,
    summary="Request password reset",
    description="Send OTP code to user's email for password reset"
)
async def reset_password_request(
    email: EmailStr,
    user_service: UserService = Depends(get_user_service),
    redis_service: RedisService = Depends(get_redis_service)
):
    """
    Send OTP code to user's email for password reset.
    """
    
    # Check if user exists
    user_doc = await user_service.get_user_by_email(email)
    if not user_doc:
        # Don't reveal if email exists for security
        return MessageResponse(
            success=True,
            message="If the email exists, a reset code has been sent"
        )
    
    # Generate OTP code
    otp_code = generate_password_reset_otp()
    
    # Store OTP in Redis with 10 minute expiration
    otp_stored = await redis_service.set_otp(email, otp_code, expire_minutes=10)
    
    if not otp_stored:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate reset code"
        )
    
    # Send OTP via email
    try:
        email_sent = send_password_reset_email(email, otp_code, "")
        if not email_sent:
            # Clean up OTP if email failed
            await redis_service.delete_otp(email)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send reset email"
            )
    except Exception as e:
        # Clean up OTP if email failed
        await redis_service.delete_otp(email)
        print(f"Failed to send reset email: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send reset email"
        )
    
    return MessageResponse(
        success=True,
        message="Reset code has been sent to your email"
    )


@router.post(
    "/reset-password-approve",
    response_model=MessageResponse,
    summary="Approve password reset",
    description="Validate OTP code and reset password"
)
async def reset_password_approve(
    reset_data: PasswordResetRequest,
    user_service: UserService = Depends(get_user_service),
    redis_service: RedisService = Depends(get_redis_service)
):
    """
    Validate OTP code and reset user password.
    """
    
    # Validate OTP format
    if not validate_password_reset_otp_format(reset_data.otp_code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OTP format"
        )
    
    # Check if user exists
    user_doc = await user_service.get_user_by_email(reset_data.email)
    if not user_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Verify OTP code (this will also delete it from Redis if valid)
    otp_valid = await redis_service.verify_otp(reset_data.email, reset_data.otp_code)
    
    if not otp_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP code"
        )
    
    # Reset password using user service
    new_password_hash = get_password_hash(reset_data.new_password)
    
    success = await user_service.update_user(
        reset_data.email, 
        {"password_hash": new_password_hash}
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password"
        )
    
    return MessageResponse(
        success=True,
        message="Password reset successfully"
    )


@router.post(
    "/change-password",
    response_model=MessageResponse,
    summary="Change password",
    description="Change user password (requires authentication)"
)
async def change_password(
    request: ChangePasswordRequest,
    current_user_email: str = Depends(get_current_user_email),
    user_service: UserService = Depends(get_user_service)
):
    """
    Change user password. Requires current password verification.
    """
    
    # Update password using service (handles validation internally)
    success = await user_service.update_password(
        current_user_email, 
        request.current_password, 
        request.new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect or user not found"
        )
    
    return MessageResponse(
        success=True,
        message="Password updated successfully"
    )