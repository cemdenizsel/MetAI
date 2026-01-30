package com.dag.nexem.features.auth.register.features

import com.dag.nexem.base.components.TextFieldState

data class RegisterViewState(
    val fullName: String = "",
    val fullNameState: TextFieldState = TextFieldState.NORMAL,
    val fullNameError: String? = null,

    val email: String = "",
    val emailState: TextFieldState = TextFieldState.NORMAL,
    val emailError: String? = null,

    val password: String = "",
    val passwordState: TextFieldState = TextFieldState.NORMAL,
    val passwordError: String? = null,

    val confirmPassword: String = "",
    val confirmPasswordState: TextFieldState = TextFieldState.NORMAL,
    val confirmPasswordError: String? = null,

    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    val isRegistrationSuccessful: Boolean = false
)