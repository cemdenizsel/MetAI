package com.dag.nexem.features.auth.login.features

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
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
import com.dag.nexem.ui.theme.Gray600
import com.dag.nexem.ui.theme.NexemTheme

@Composable
fun LoginScreen(
    navigationController: NavHostController,
    viewModel: LoginViewModel = hiltViewModel()
) {
    val uiState by viewModel.viewState.collectAsState()

    // Navigate to home on successful login
    LaunchedEffect(uiState.isLoginSuccessful) {
        if (uiState.isLoginSuccessful) {
            navigationController.navigate(Destination.HomeScreen) {
                popUpTo(0) { inclusive = true }
            }
        }
    }

    // Show error message if login fails
    LaunchedEffect(uiState.errorMessage) {
        uiState.errorMessage?.let { error ->
            // You can show a Snackbar or Toast here if needed
            // For now, the error will be displayed in the UI
        }
    }

    AuthSurface(
        headerTitle = stringResource(R.string.login_page_title),
        headerSubtitle = stringResource(R.string.login_page_subtitle)
    ) {
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.SpaceEvenly
        ) {
            Column(
                verticalArrangement = Arrangement.spacedBy(Dimen.SpaceMd)
            ) {
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

                Column(
                    verticalArrangement = Arrangement.spacedBy(Dimen.SpaceSm)
                ) {
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

                    TextButton(
                        {
                            navigationController.navigate(Destination.PasswordActionScreen)
                        }
                    ) {
                        Text(
                            stringResource(R.string.login_page_forgot_password_button),
                            style = MaterialTheme.typography.labelSmall.copy(
                                color = MaterialTheme.colorScheme.primary
                            )
                        )
                    }
                }

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

                RedirectionButton(
                    stringResource(R.string.login_redirection_button_text),
                    Gray400
                ) {
                    navigationController.navigate(Destination.RegisterScreen)
                }

                // Login Button with loading state
                if (uiState.isLoading) {
                    CircularProgressIndicator(
                        modifier = Modifier.align(Alignment.CenterHorizontally)
                    )
                } else {
                    CustomButton(
                        text = stringResource(R.string.login_page_button_text)
                    ) {
                        viewModel.validateAndLogin()
                    }
                }
            }
        }
    }
}

@Composable
@Preview
fun LoginScreenPreview() {
    NexemTheme {
        LoginScreen(rememberNavController())
    }
}