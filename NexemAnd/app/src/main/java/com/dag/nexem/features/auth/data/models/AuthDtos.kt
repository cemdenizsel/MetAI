package com.dag.nexem.features.auth.data.models

import com.google.gson.annotations.SerializedName

// Request DTOs

data class UserRegistrationRequestDto(
    @SerializedName("username")
    val username: String,
    @SerializedName("email")
    val email: String,
    @SerializedName("password")
    val password: String
)

data class UserLoginRequestDto(
    @SerializedName("email")
    val email: String,
    @SerializedName("password")
    val password: String
)

data class PasswordResetRequestDto(
    @SerializedName("email")
    val email: String,
    @SerializedName("otp_code")
    val otpCode: String,
    @SerializedName("new_password")
    val newPassword: String
)

data class ChangePasswordRequestDto(
    @SerializedName("current_password")
    val currentPassword: String,
    @SerializedName("new_password")
    val newPassword: String
)

// Response DTOs

data class TokenResponseDto(
    @SerializedName("access_token")
    val accessToken: String,
    @SerializedName("token_type")
    val tokenType: String,
    @SerializedName("expires_in")
    val expiresIn: Int,
    @SerializedName("user")
    val user: UserResponseDto
)

data class UserResponseDto(
    @SerializedName("username")
    val username: String,
    @SerializedName("email")
    val email: String,
    @SerializedName("package")
    val packageName: String,
    @SerializedName("created_at")
    val createdAt: String,
    @SerializedName("updated_at")
    val updatedAt: String
)

data class MessageResponseDto(
    @SerializedName("message")
    val message: String
)
