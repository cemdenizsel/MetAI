package com.dag.nexem.features.auth.register.features

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.rememberNavController
import com.dag.nexem.R
import com.dag.nexem.base.components.CustomButton
import com.dag.nexem.base.components.CustomTextField
import com.dag.nexem.base.components.TextFieldType
import com.dag.nexem.base.navigation.Destination
import com.dag.nexem.features.auth.common.components.AuthSurface
import com.dag.nexem.features.auth.common.components.RedirectionButton
import com.dag.nexem.ui.theme.Dimen
import com.dag.nexem.ui.theme.Gray400
import com.dag.nexem.ui.theme.NexemTheme

@Composable
fun RegisterScreen(
    navigationController: NavHostController,
    viewModel: RegisterViewModel = hiltViewModel()
) {
    val uiState by viewModel.viewState.collectAsState()

    // Navigate to home on successful registration
    LaunchedEffect(uiState.isRegistrationSuccessful) {
        if (uiState.isRegistrationSuccessful) {
            navigationController.navigate(Destination.HomeScreen) {
                popUpTo(0) { inclusive = true }
            }
        }
    }

    // Show error message if registration fails
    LaunchedEffect(uiState.errorMessage) {
        uiState.errorMessage?.let { error ->
            // You can show a Snackbar or Toast here if needed
            // For now, the error will be displayed in the UI
        }
    }

    AuthSurface(
        headerTitle = stringResource(R.string.register_page_title),
        headerSubtitle = stringResource(R.string.register_page_subtitle)
    ) {
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.SpaceEvenly
        ) {
            Column(
                verticalArrangement = Arrangement.spacedBy(Dimen.SpaceMd)
            ) {
                // Full Name Field
                CustomTextField(
                    value = uiState.fullName,
                    onValueChange = viewModel::onFullNameChange,
                    type = TextFieldType.TEXT,
                    label = stringResource(R.string.fullname_field),
                    placeholder = stringResource(R.string.fullname_field),
                    state = uiState.fullNameState,
                    errorMessage = uiState.fullNameError
                )

                // Email Field
                CustomTextField(
                    value = uiState.email,
                    onValueChange = viewModel::onEmailChange,
                    type = TextFieldType.TEXT,
                    label = stringResource(R.string.email_field),
                    placeholder = stringResource(R.string.email_field),
                    state = uiState.emailState,
                    errorMessage = uiState.emailError
                )

                // Password Field
                CustomTextField(
                    value = uiState.password,
                    onValueChange = viewModel::onPasswordChange,
                    type = TextFieldType.PASSWORD,
                    label = stringResource(R.string.password),
                    placeholder = stringResource(R.string.password),
                    state = uiState.passwordState,
                    errorMessage = uiState.passwordError
                )

                // Confirm Password Field
                CustomTextField(
                    value = uiState.confirmPassword,
                    onValueChange = viewModel::onConfirmPasswordChange,
                    type = TextFieldType.PASSWORD,
                    label = stringResource(R.string.password_reply),
                    placeholder = stringResource(R.string.password_reply),
                    state = uiState.confirmPasswordState,
                    errorMessage = uiState.confirmPasswordError
                )


            }

            Column(
                verticalArrangement = Arrangement.spacedBy(Dimen.SpaceXl)
            ) {
                // Show error message if present
                uiState.errorMessage?.let { error ->
                    Text(
                        text = error,
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodySmall
                    )
                }

                RedirectionButton(stringResource(R.string.register_redirection_button_text),
                    Gray400
                ) {
                    navigationController.navigate(Destination.LoginScreen)
                }

                // Register Button with loading state
                if (uiState.isLoading) {
                    CircularProgressIndicator(
                        modifier = Modifier.align(Alignment.CenterHorizontally)
                    )
                } else {
                    CustomButton(
                        text = stringResource(R.string.register_page_button_text),
                    ) {
                        viewModel.validateAndRegister()
                    }
                }
            }

        }
    }
}

@Composable
@Preview
fun RegisterScreenPreview() {
    NexemTheme {
        RegisterScreen(rememberNavController())
    }
}