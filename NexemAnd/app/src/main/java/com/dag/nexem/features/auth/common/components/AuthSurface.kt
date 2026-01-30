package com.dag.nexem.features.auth.common.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AccountCircle
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.tooling.preview.Preview
import com.dag.nexem.base.components.CustomButton
import com.dag.nexem.base.components.CustomCard
import com.dag.nexem.base.components.CustomTextField
import com.dag.nexem.base.components.TextFieldState
import com.dag.nexem.ui.theme.Dimen
import com.dag.nexem.ui.theme.NexemTheme

@Composable
fun AuthSurface(
    headerTitle: String,
    headerSubtitle: String,
    icon: ImageVector = Icons.Filled.AccountCircle,
    content: @Composable () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.onSecondaryContainer),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.size(Dimen.SpaceXl))
        CustomCard(
            icon = icon,
            textColor = MaterialTheme.colorScheme.onPrimary,
            title = headerTitle,
            subtitle = headerSubtitle
        )
        Spacer(modifier = Modifier.size(Dimen.SpaceLg))
        Column(
            modifier = Modifier
                .fillMaxSize()
                .clip(RoundedCornerShape(topEnd = Dimen.RadiusLg, topStart = Dimen.RadiusLg))
                .background(MaterialTheme.colorScheme.surface)
                .padding(Dimen.PaddingMd)
        ) {
            content()
        }
    }
}


@Preview
@Composable
fun PreviewAuthHeader() {
    NexemTheme {
        Surface {
            AuthSurface(
                "Welcome back",
                "Sign in to continue your emotion analysis journey"
            ) {
                Column(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.SpaceEvenly
                ) {
                    Column {
                        CustomTextField(
                            value = "",
                            onValueChange = { },
                            label = "Normal State",
                            placeholder = "Enter text...",
                            helperText = "This is helper text",
                            state = TextFieldState.NORMAL,
                            modifier = Modifier.padding(bottom = Dimen.SpaceMd)
                        )

                        CustomTextField(
                            value = "Focused text",
                            onValueChange = { },
                            label = "Focused State",
                            placeholder = "Enter text...",
                            state = TextFieldState.FOCUSED,
                            modifier = Modifier.padding(bottom = Dimen.SpaceMd)
                        )
                    }
                    CustomButton(text = "Sign In") {}
                }
            }
        }
    }
}