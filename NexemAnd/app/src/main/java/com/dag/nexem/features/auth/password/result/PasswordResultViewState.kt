package com.dag.nexem.features.auth.password.result

data class PasswordResultViewState(
    val email: String = "",
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    val isEmailResent: Boolean = false
)