package com.dag.nexem.features.auth.register.features

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
class RegisterViewModel @Inject constructor(
    private val authRepository: AuthRepository
) : ViewModel() {

    private val _viewState = MutableStateFlow(RegisterViewState())
    val viewState: StateFlow<RegisterViewState> = _viewState.asStateFlow()

    fun onFullNameChange(value: String) {
        _viewState.update { it.copy(
            fullName = value,
            fullNameState = TextFieldState.NORMAL,
            fullNameError = null
        ) }
    }

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

    fun onConfirmPasswordChange(value: String) {
        _viewState.update { it.copy(
            confirmPassword = value,
            confirmPasswordState = TextFieldState.NORMAL,
            confirmPasswordError = null
        ) }
    }

    fun validateAndRegister() {
        var hasError = false

        // Validate full name
        if (_viewState.value.fullName.isBlank()) {
            _viewState.update { it.copy(
                fullNameState = TextFieldState.ERROR,
                fullNameError = "Full name is required"
            ) }
            hasError = true
        }

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
        } else if (_viewState.value.password.length < 6) {
            _viewState.update { it.copy(
                passwordState = TextFieldState.ERROR,
                passwordError = "Password must be at least 6 characters"
            ) }
            hasError = true
        }

        // Validate confirm password
        if (_viewState.value.confirmPassword.isBlank()) {
            _viewState.update { it.copy(
                confirmPasswordState = TextFieldState.ERROR,
                confirmPasswordError = "Please confirm your password"
            ) }
            hasError = true
        } else if (_viewState.value.password != _viewState.value.confirmPassword) {
            _viewState.update { it.copy(
                confirmPasswordState = TextFieldState.ERROR,
                confirmPasswordError = "Passwords do not match"
            ) }
            hasError = true
        }

        if (!hasError) {
            // Proceed with registration
            register()
        }
    }

    private fun register() {
        _viewState.update { it.copy(
            isLoading = true,
            errorMessage = null,
            isRegistrationSuccessful = false
        ) }

        viewModelScope.launch {
            when (val result = authRepository.register(
                username = _viewState.value.fullName,
                email = _viewState.value.email,
                password = _viewState.value.password
            )) {
                is AuthResult.Success -> {
                    _viewState.update { it.copy(
                        isLoading = false,
                        isRegistrationSuccessful = true,
                        errorMessage = null
                    ) }
                }
                is AuthResult.Error -> {
                    _viewState.update { it.copy(
                        isLoading = false,
                        isRegistrationSuccessful = false,
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