package com.dag.nexem.features.auth.domain.repository

import com.dag.nexem.features.auth.domain.models.AuthResult
import com.dag.nexem.features.auth.domain.models.AuthToken
import com.dag.nexem.features.auth.domain.models.User

interface AuthRepository {

    suspend fun register(
        username: String,
        email: String,
        password: String
    ): AuthResult<AuthToken>

    suspend fun login(
        email: String,
        password: String
    ): AuthResult<AuthToken>

    suspend fun getProfile(): AuthResult<User>

    suspend fun requestPasswordReset(email: String): AuthResult<String>

    suspend fun approvePasswordReset(
        email: String,
        otpCode: String,
        newPassword: String
    ): AuthResult<String>

    suspend fun changePassword(
        currentPassword: String,
        newPassword: String
    ): AuthResult<String>
}
