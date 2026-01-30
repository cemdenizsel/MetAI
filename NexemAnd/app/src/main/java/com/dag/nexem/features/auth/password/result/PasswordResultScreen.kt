package com.dag.nexem.features.auth.password.result

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
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
import com.dag.nexem.base.navigation.Destination
import com.dag.nexem.base.components.CustomButton
import com.dag.nexem.base.components.SecondaryCustomButton
import com.dag.nexem.features.auth.common.components.AuthSurface
import com.dag.nexem.ui.theme.Dimen
import com.dag.nexem.ui.theme.Gray100
import com.dag.nexem.ui.theme.Green100
import com.dag.nexem.ui.theme.Green600
import com.dag.nexem.ui.theme.NexemTheme

@Composable
fun PasswordResultScreen(
    navigationController: NavHostController? = null,
    viewModel: PasswordResultViewModel = hiltViewModel()
) {
    val uiState by viewModel.viewState.collectAsState()

    AuthSurface(
        stringResource(R.string.forgot_password_result_page_title),
        stringResource(R.string.forgot_password_result_page_subtitle),
        Icons.Default.Check
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = Dimen.PaddingMd),
            verticalArrangement = Arrangement.spacedBy(Dimen.SpaceMd)
        ) {
            // Success message box with email
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        color = Green100,
                        shape = RoundedCornerShape(Dimen.RadiusMd)
                    )
                    .padding(Dimen.PaddingLg),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = stringResource(R.string.forgot_password_result_page_success_message),
                    style = MaterialTheme.typography.bodyLarge,
                    color = Green600
                )
                Text(
                    text = uiState.email,
                    style = MaterialTheme.typography.bodyLarge,
                    color = Green600
                )
            }

            // Info message
            Text(
                text = stringResource(R.string.forgot_password_result_page_info_message),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface
            )


            // Didn't receive email section
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(
                        color = Gray100,
                        shape = RoundedCornerShape(Dimen.RadiusMd)
                    )
                    .padding(Dimen.PaddingLg),
                verticalArrangement = Arrangement.spacedBy(Dimen.SpaceSm)
            ) {
                Text(
                    text = stringResource(R.string.forgot_password_result_page_failed_email_info),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface
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

            // Buttons
            Column(
                verticalArrangement = Arrangement.spacedBy(Dimen.SpaceMd)
            ) {
                // Resend email button with loading state
                if (uiState.isLoading) {
                    CircularProgressIndicator(
                        modifier = Modifier.align(Alignment.CenterHorizontally)
                    )
                } else {
                    SecondaryCustomButton(
                        text = stringResource(R.string.forgot_password_result_page_secondary_button_text)
                    ) {
                        viewModel.onResendEmail()
                    }
                }

                // Back to login button
                CustomButton(
                    text = stringResource(R.string.forgot_password_result_page_primary_button_text)
                ) {
                    navigationController?.navigate(Destination.LoginScreen) {
                        popUpTo(0) { inclusive = true }
                    }
                }
            }

            Spacer(modifier = Modifier.height(Dimen.SpaceMd))
        }
    }
}

@Composable
@Preview
fun PasswordResultScreenPreview() {
    NexemTheme {
        PasswordResultScreen(rememberNavController())
    }
}
