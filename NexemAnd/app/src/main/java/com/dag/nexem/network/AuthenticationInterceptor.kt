package com.dag.nexem.network

import okhttp3.Interceptor
import okhttp3.Response
import javax.inject.Inject

class AuthenticationInterceptor @Inject constructor(
    private val tokenManager: TokenManager
) : Interceptor {

    override fun intercept(chain: Interceptor.Chain): Response {
        val originalRequest = chain.request()

        // Check if the request URL requires authentication
        val requiresAuth = originalRequest.url.encodedPath.requiresAuthentication()

        return if (requiresAuth) {
            val token = tokenManager.getToken()
            if (token != null) {
                val authenticatedRequest = originalRequest.newBuilder()
                    .header("Authorization", "Bearer $token")
                    .build()
                chain.proceed(authenticatedRequest)
            } else {
                // Proceed without token if not available
                chain.proceed(originalRequest)
            }
        } else {
            // Proceed without authentication
            chain.proceed(originalRequest)
        }
    }

    private fun String.requiresAuthentication(): Boolean {
        // Define paths that DON'T require authentication
        val publicPaths = listOf(
            "/auth/login",
            "/auth/register",
            "/auth/reset-password-request",
            "/auth/reset-password-approve",
            "/"
        )

        // Return true if the path is NOT in the public paths list
        return !publicPaths.any { this.contains(it) }
    }
}
