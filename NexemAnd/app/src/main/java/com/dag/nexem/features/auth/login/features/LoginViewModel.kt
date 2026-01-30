package com.dag.nexem.features.auth.login.features

import android.util.Patterns
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dag.nexem.base.components.TextFieldState
import com.dag.nexem.features.auth.domain.models.AuthResult
import com.dag.nexem.features.auth.domain.repository.AuthRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class LoginViewModel @Inject constructor(
    private val authRepository: AuthRepository
) : ViewModel() {

    private val _viewState = MutableStateFlow(LoginViewState())
    val viewState: StateFlow<LoginViewState> = _viewState.asStateFlow()

    fun onEmailChange(value: String) {
        _viewState.update { it.copy(
            email = value,
            emailState = TextFieldState.NORMAL,
            emailError = null
        ) }
    }

    fun onPasswordChange(value: String) {
        _viewState.update { it.copy(
            password = value,
            passwordState = TextFieldState.NORMAL,
            passwordError = null
        ) }
    }

    fun validateAndLogin() {
        var hasError = false

        // Validate email
        if (_viewState.value.email.isBlank()) {
            _viewState.update { it.copy(
                emailState = TextFieldState.ERROR,
                emailError = "Email is required"
            ) }
            hasError = true
        } else if (!isValidEmail(_viewState.value.email)) {
            _viewState.update { it.copy(
                emailState = TextFieldState.ERROR,
                emailError = "Invalid email format"
            ) }
            hasError = true
        }

        // Validate password
        if (_viewState.value.password.isBlank()) {
            _viewState.update { it.copy(
                passwordState = TextFieldState.ERROR,
                passwordError = "Password is required"
            ) }
            hasError = true
        }

        if (!hasError) {
            // Proceed with login
            login()
        }
    }

    private fun login() {
        _viewState.update { it.copy(
            isLoading = true,
            errorMessage = null,
            isLoginSuccessful = false
        ) }

        viewModelScope.launch {
            when (val result = authRepository.login(_viewState.value.email, _viewState.value.password)) {
                is AuthResult.Success -> {
                    _viewState.update { it.copy(
                        isLoading = false,
                        isLoginSuccessful = true,
                        errorMessage = null
                    ) }
                }
                is AuthResult.Error -> {
                    _viewState.update { it.copy(
                        isLoading = false,
                        isLoginSuccessful = false,
                        errorMessage = result.message
                    ) }
                }
                is AuthResult.Loading -> {
                    _viewState.update { it.copy(isLoading = true) }
                }
            }
        }
    }

    fun clearError() {
        _viewState.update { it.copy(errorMessage = null) }
    }

    private fun isValidEmail(email: String): Boolean {
        return Patterns.EMAIL_ADDRESS.matcher(email).matches()
    }
}