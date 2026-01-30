package com.dag.nexem.features.auth.password.action

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Email
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
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.rememberNavController
import com.dag.nexem.R
import com.dag.nexem.base.navigation.Destination
import com.dag.nexem.base.components.CustomButton
import com.dag.nexem.base.components.CustomTextField
import com.dag.nexem.base.components.TextFieldType
import com.dag.nexem.features.auth.common.components.AuthSurface
import com.dag.nexem.ui.theme.Dimen
import com.dag.nexem.ui.theme.NexemTheme


@Composable
fun PasswordActionScreen(
    navigationController: NavHostController? = null,
    viewModel: PasswordActionViewModel = hiltViewModel()
) {
    val uiState by viewModel.viewState.collectAsState()

    // Navigate to PasswordResultScreen when reset email is sent
    LaunchedEffect(uiState.isResetEmailSent) {
        if (uiState.isResetEmailSent) {
            navigationController?.navigate(Destination.PasswordResultScreen)
        }
    }

    AuthSurface(
        stringResource(R.string.forgot_password_page_title),
        stringResource(R.string.forgot_password_page_subtitle),
        Icons.Default.Email
    ) {
        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(Dimen.SpaceXl),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Column {
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
                Spacer(modifier = Modifier.size(Dimen.SpaceSm))
                Text(
                    stringResource(R.string.forgot_password_page_hint),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                )
            }

            // Show error message if present
            uiState.errorMessage?.let { error ->
                Text(
                    text = error,
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall
                )
            }

            // Send Reset Email Button with loading state
            if (uiState.isLoading) {
                CircularProgressIndicator()
            } else {
                CustomButton(
                    text = stringResource(R.string.forgot_password_page_reset_link_button_text)
                ) {
                    viewModel.validateAndSendResetEmail()
                }
            }

            Text(
                stringResource(R.string.forgot_password_page_remember_button),
                style = MaterialTheme.typography.labelMedium
                    .copy(MaterialTheme.colorScheme.onPrimaryContainer),
                modifier = Modifier.clickable{
                    navigationController?.navigateUp()
                }
            )
        }
    }
}

@Composable
@Preview
fun PasswordActionScreenPreview() {
    NexemTheme {
        PasswordActionScreen(rememberNavController())
    }
}