package com.dag.nexem.features.auth.password.result

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
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
class PasswordResultViewModel @Inject constructor(
    private val authRepository: AuthRepository
) : ViewModel() {

    private val _viewState = MutableStateFlow(PasswordResultViewState())
    val viewState: StateFlow<PasswordResultViewState> = _viewState.asStateFlow()

    fun setEmail(email: String) {
        _viewState.update { it.copy(email = email) }
    }

    fun onResendEmail() {
        _viewState.update { it.copy(
            isLoading = true,
            errorMessage = null,
            isEmailResent = false
        ) }

        viewModelScope.launch {
            when (val result = authRepository.requestPasswordReset(_viewState.value.email)) {
                is AuthResult.Success -> {
                    _viewState.update { it.copy(
                        isLoading = false,
                        isEmailResent = true,
                        errorMessage = null
                    ) }
                }
                is AuthResult.Error -> {
                    _viewState.update { it.copy(
                        isLoading = false,
                        isEmailResent = false,
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

    fun onBackToLogin() {
        // TODO: Navigate back to login screen
    }
}
