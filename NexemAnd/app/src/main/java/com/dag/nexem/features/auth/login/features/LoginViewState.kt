package com.dag.nexem.features.auth.login.features

import com.dag.nexem.base.components.TextFieldState

data class LoginViewState(
    val email: String = "",
    val emailState: TextFieldState = TextFieldState.NORMAL,
    val emailError: String? = null,

    val password: String = "",
    val passwordState: TextFieldState = TextFieldState.NORMAL,
    val passwordError: String? = null,

    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    val isLoginSuccessful: Boolean = false
)