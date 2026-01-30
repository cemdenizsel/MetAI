package com.dag.nexem.features.auth.data.remote

import com.dag.nexem.features.auth.data.models.ChangePasswordRequestDto
import com.dag.nexem.features.auth.data.models.MessageResponseDto
import com.dag.nexem.features.auth.data.models.PasswordResetRequestDto
import com.dag.nexem.features.auth.data.models.TokenResponseDto
import com.dag.nexem.features.auth.data.models.UserLoginRequestDto
import com.dag.nexem.features.auth.data.models.UserRegistrationRequestDto
import com.dag.nexem.features.auth.data.models.UserResponseDto
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Query

interface AuthApiService {

    @POST("auth/register")
    suspend fun register(
        @Body request: UserRegistrationRequestDto
    ): Response<TokenResponseDto>

    @POST("auth/login")
    suspend fun login(
        @Body request: UserLoginRequestDto
    ): Response<TokenResponseDto>

    @GET("auth/profile")
    suspend fun getProfile(): Response<UserResponseDto>

    @POST("auth/reset-password-request")
    suspend fun requestPasswordReset(
        @Query("email") email: String
    ): Response<MessageResponseDto>

    @POST("auth/reset-password-approve")
    suspend fun approvePasswordReset(
        @Body request: PasswordResetRequestDto
    ): Response<MessageResponseDto>

    @POST("auth/change-password")
    suspend fun changePassword(
        @Body request: ChangePasswordRequestDto
    ): Response<MessageResponseDto>
}
