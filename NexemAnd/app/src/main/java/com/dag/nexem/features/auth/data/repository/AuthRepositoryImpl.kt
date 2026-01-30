package com.dag.nexem.features.auth.data.repository

import com.dag.nexem.features.auth.data.models.ChangePasswordRequestDto
import com.dag.nexem.features.auth.data.models.PasswordResetRequestDto
import com.dag.nexem.features.auth.data.models.UserLoginRequestDto
import com.dag.nexem.features.auth.data.models.UserRegistrationRequestDto
import com.dag.nexem.features.auth.data.remote.AuthApiService
import com.dag.nexem.features.auth.domain.models.AuthResult
import com.dag.nexem.features.auth.domain.models.AuthToken
import com.dag.nexem.features.auth.domain.models.User
import com.dag.nexem.features.auth.domain.repository.AuthRepository
import com.dag.nexem.network.TokenManager
import javax.inject.Inject

class AuthRepositoryImpl @Inject constructor(
    private val authApiService: AuthApiService,
    private val tokenManager: TokenManager
) : AuthRepository {

    override suspend fun register(
        username: String,
        email: String,
        password: String
    ): AuthResult<AuthToken> {
        return try {
            val response = authApiService.register(
                UserRegistrationRequestDto(username, email, password)
            )

            if (response.isSuccessful && response.body() != null) {
                val tokenDto = response.body()!!
                // Save token
                tokenManager.saveToken(tokenDto.accessToken)

                AuthResult.Success(
                    AuthToken(
                        accessToken = tokenDto.accessToken,
                        tokenType = tokenDto.tokenType,
                        expiresIn = tokenDto.expiresIn,
                        user = User(
                            username = tokenDto.user.username,
                            email = tokenDto.user.email,
                            packageName = tokenDto.user.packageName,
                            createdAt = tokenDto.user.createdAt,
                            updatedAt = tokenDto.user.updatedAt
                        )
                    )
                )
            } else {
                AuthResult.Error(
                    message = response.message() ?: "Registration failed",
                    code = response.code()
                )
            }
        } catch (e: Exception) {
            AuthResult.Error(message = e.message ?: "Unknown error occurred")
        }
    }

    override suspend fun login(email: String, password: String): AuthResult<AuthToken> {
        return try {
            val response = authApiService.login(
                UserLoginRequestDto(email, password)
            )

            if (response.isSuccessful && response.body() != null) {
                val tokenDto = response.body()!!
                // Save token
                tokenManager.saveToken(tokenDto.accessToken)

                AuthResult.Success(
                    AuthToken(
                        accessToken = tokenDto.accessToken,
                        tokenType = tokenDto.tokenType,
                        expiresIn = tokenDto.expiresIn,
                        user = User(
                            username = tokenDto.user.username,
                            email = tokenDto.user.email,
                            packageName = tokenDto.user.packageName,
                            createdAt = tokenDto.user.createdAt,
                            updatedAt = tokenDto.user.updatedAt
                        )
                    )
                )
            } else {
                AuthResult.Error(
                    message = response.message() ?: "Login failed",
                    code = response.code()
                )
            }
        } catch (e: Exception) {
            AuthResult.Error(message = e.message ?: "Unknown error occurred")
        }
    }

    override suspend fun getProfile(): AuthResult<User> {
        return try {
            val response = authApiService.getProfile()

            if (response.isSuccessful && response.body() != null) {
                val userDto = response.body()!!
                AuthResult.Success(
                    User(
                        username = userDto.username,
                        email = userDto.email,
                        packageName = userDto.packageName,
                        createdAt = userDto.createdAt,
                        updatedAt = userDto.updatedAt
                    )
                )
            } else {
                AuthResult.Error(
                    message = response.message() ?: "Failed to fetch profile",
                    code = response.code()
                )
            }
        } catch (e: Exception) {
            AuthResult.Error(message = e.message ?: "Unknown error occurred")
        }
    }

    override suspend fun requestPasswordReset(email: String): AuthResult<String> {
        return try {
            val response = authApiService.requestPasswordReset(email)

            if (response.isSuccessful && response.body() != null) {
                AuthResult.Success(response.body()!!.message)
            } else {
                AuthResult.Error(
                    message = response.message() ?: "Failed to request password reset",
                    code = response.code()
                )
            }
        } catch (e: Exception) {
            AuthResult.Error(message = e.message ?: "Unknown error occurred")
        }
    }

    override suspend fun approvePasswordReset(
        email: String,
        otpCode: String,
        newPassword: String
    ): AuthResult<String> {
        return try {
            val response = authApiService.approvePasswordReset(
                PasswordResetRequestDto(email, otpCode, newPassword)
            )

            if (response.isSuccessful && response.body() != null) {
                AuthResult.Success(response.body()!!.message)
            } else {
                AuthResult.Error(
                    message = response.message() ?: "Failed to reset password",
                    code = response.code()
                )
            }
        } catch (e: Exception) {
            AuthResult.Error(message = e.message ?: "Unknown error occurred")
        }
    }

    override suspend fun changePassword(
        currentPassword: String,
        newPassword: String
    ): AuthResult<String> {
        return try {
            val response = authApiService.changePassword(
                ChangePasswordRequestDto(currentPassword, newPassword)
            )

            if (response.isSuccessful && response.body() != null) {
                AuthResult.Success(response.body()!!.message)
            } else {
                AuthResult.Error(
                    message = response.message() ?: "Failed to change password",
                    code = response.code()
                )
            }
        } catch (e: Exception) {
            AuthResult.Error(message = e.message ?: "Unknown error occurred")
        }
    }
}
