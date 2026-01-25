from pydantic import BaseModel, Field, EmailStr
from enum import Enum

class Packages(Enum):
    Basic = 1
    Premium = 2

class UserModel(BaseModel):
    username: str = Field(description= "Username")
    password: str = Field(description= "Password")
    email: str = Field(description= "Email")
    package: str = Field(description= "Package")


# Request/Response ai_models
class UserRegistrationRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., min_length=6, max_length=100, description="Password")


class UserLoginRequest(BaseModel):
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class UserResponse(BaseModel):
    username: str
    email: str
    package: str
    created_at: float
    updated_at: float


class MessageResponse(BaseModel):
    success: bool
    message: str


class PasswordResetRequest(BaseModel):
    email: EmailStr = Field(..., description="Email address")
    otp_code: str = Field(..., min_length=6, max_length=6, description="6-digit OTP code")
    new_password: str = Field(..., min_length=6, max_length=100, description="New password")


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=6, max_length=100, description="New password")
