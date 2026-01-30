package com.dag.nexem.features.auth.password.action

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
class PasswordActionViewModel @Inject constructor(
    private val authRepository: AuthRepository
) : ViewModel() {

    private val _viewState = MutableStateFlow(PasswordActionViewState())
    val viewState: StateFlow<PasswordActionViewState> = _viewState.asStateFlow()

    fun onEmailChange(value: String) {
        _viewState.update { it.copy(
            email = value,
            emailState = TextFieldState.NORMAL,
            emailError = null
        ) }
    }

    fun validateAndSendResetEmail() {
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

        if (!hasError) {
            // Proceed with sending reset email
            sendResetEmail()
        }
    }

    private fun sendResetEmail() {
        _viewState.update { it.copy(
            isLoading = true,
            errorMessage = null,
            isResetEmailSent = false
        ) }

        viewModelScope.launch {
            when (val result = authRepository.requestPasswordReset(_viewState.value.email)) {
                is AuthResult.Success -> {
                    _viewState.update { it.copy(
                        isLoading = false,
                        isResetEmailSent = true,
                        errorMessage = null
                    ) }
                }
                is AuthResult.Error -> {
                    _viewState.update { it.copy(
                        isLoading = false,
                        isResetEmailSent = false,
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
        return android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches()
    }
}