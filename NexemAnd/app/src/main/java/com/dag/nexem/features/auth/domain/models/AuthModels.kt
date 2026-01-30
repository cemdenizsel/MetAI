package com.dag.nexem.features.auth.domain.models

// Domain models (Business logic entities)

data class User(
    val username: String,
    val email: String,
    val packageName: String,
    val createdAt: String,
    val updatedAt: String
)

data class AuthToken(
    val accessToken: String,
    val tokenType: String,
    val expiresIn: Int,
    val user: User
)

sealed class AuthResult<out T> {
    data class Success<T>(val data: T) : AuthResult<T>()
    data class Error(val message: String, val code: Int? = null) : AuthResult<Nothing>()
    data object Loading : AuthResult<Nothing>()
}
