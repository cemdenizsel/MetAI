package com.dag.nexem.features.auth.password.action

import com.dag.nexem.base.components.TextFieldState

data class PasswordActionViewState(
    val email: String = "",
    val emailState: TextFieldState = TextFieldState.NORMAL,
    val emailError: String? = null,
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    val isResetEmailSent: Boolean = false
)