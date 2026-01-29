package com.dag.nexem.base.components

import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.VisibilityOff
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.dag.nexem.ui.theme.*

enum class TextFieldState {
    NORMAL,
    FOCUSED,
    ERROR,
    SUCCESS,
    DISABLED
}

enum class TextFieldType {
    TEXT,
    NUMBER,
    PASSWORD
}

@Composable
fun CustomTextField(
    value: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    type: TextFieldType = TextFieldType.TEXT,
    state: TextFieldState = TextFieldState.NORMAL,
    label: String? = null,
    placeholder: String? = null,
    helperText: String? = null,
    errorMessage: String? = null,
    successMessage: String? = null,
    leadingIcon: @Composable (() -> Unit)? = null,
    trailingIcon: @Composable (() -> Unit)? = null,
    keyboardOptions: KeyboardOptions = KeyboardOptions.Default,
    keyboardActions: KeyboardActions = KeyboardActions.Default,
    visualTransformation: VisualTransformation = VisualTransformation.None,
    singleLine: Boolean = true,
    maxLines: Int = if (singleLine) 1 else Int.MAX_VALUE,
    minLines: Int = 1,
    enabled: Boolean = true
) {
    var passwordVisible by remember { mutableStateOf(false) }

    val isError = state == TextFieldState.ERROR
    val isSuccess = state == TextFieldState.SUCCESS

    // Determine keyboard type based on field type
    val effectiveKeyboardOptions = when (type) {
        TextFieldType.NUMBER -> keyboardOptions.copy(keyboardType = KeyboardType.Number)
        TextFieldType.PASSWORD -> keyboardOptions.copy(keyboardType = KeyboardType.Password)
        else -> keyboardOptions
    }

    // Determine visual transformation based on field type
    val effectiveVisualTransformation = when (type) {
        TextFieldType.PASSWORD -> if (passwordVisible) VisualTransformation.None else PasswordVisualTransformation()
        else -> visualTransformation
    }

    // Handle value change with number filtering
    val effectiveOnValueChange: (String) -> Unit = { newValue ->
        when (type) {
            TextFieldType.NUMBER -> {
                // Only allow digits
                if (newValue.isEmpty() || newValue.all { it.isDigit() }) {
                    onValueChange(newValue)
                }
            }

            else -> onValueChange(newValue)
        }
    }

    // Determine trailing icon
    val effectiveTrailingIcon: (@Composable () -> Unit)? = when {
        type == TextFieldType.PASSWORD -> {
            {
                IconButton(onClick = { passwordVisible = !passwordVisible }) {
                    Icon(
                        imageVector = if (passwordVisible) Icons.Filled.Visibility else Icons.Filled.VisibilityOff,
                        contentDescription = if (passwordVisible) "Hide password" else "Show password",
                        tint = Gray400
                    )
                }
            }
        }

        else -> trailingIcon
    }

    // Determine supporting text based on state
    val supportingTextContent = when {
        isError && errorMessage != null -> errorMessage
        isSuccess && successMessage != null -> successMessage
        helperText != null -> helperText
        else -> null
    }

    Column(
        modifier = Modifier
            .padding(vertical = Dimen.PaddingXs)
    ) {
        // Label
        if (label != null) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelMedium,
                color = when (state) {
                    TextFieldState.ERROR -> Red600
                    TextFieldState.SUCCESS -> Green600
                    TextFieldState.FOCUSED -> Indigo600
                    TextFieldState.DISABLED -> Gray400
                    else -> Gray600
                },
                modifier = Modifier.padding(bottom = Dimen.PaddingXs)
            )
        }

        // Text Field
        TextField(
            value = value,
            onValueChange = effectiveOnValueChange,
            modifier = modifier
                .fillMaxWidth()
                .border(
                    width = 1.dp,
                    color = Color.LightGray,
                    shape = RoundedCornerShape(Dimen.RadiusSm)
                ),
            enabled = enabled && state != TextFieldState.DISABLED,
            label = if (label != null) {
                { Text(label) }
            } else null,
            placeholder = if (placeholder != null) {
                { Text(placeholder) }
            } else null,
            leadingIcon = leadingIcon,
            trailingIcon = effectiveTrailingIcon,
            isError = state == TextFieldState.ERROR,
            visualTransformation = effectiveVisualTransformation,
            keyboardOptions = effectiveKeyboardOptions,
            keyboardActions = keyboardActions,
            singleLine = singleLine,
            maxLines = maxLines,
            minLines = minLines,
            colors = TextFieldDefaults.colors(
                focusedContainerColor = Color.Transparent,
                unfocusedContainerColor = Color.Transparent,
                disabledContainerColor = Color.Transparent,
                errorContainerColor = Color.Transparent,
                focusedIndicatorColor = Color.Transparent,
                unfocusedIndicatorColor = Color.Transparent,
                disabledIndicatorColor = Color.Transparent,
                errorIndicatorColor = Color.Transparent,
                focusedTextColor = Gray900,
                unfocusedTextColor = Gray900,
                disabledTextColor = Gray400,
                errorTextColor = Gray900,
                focusedPlaceholderColor = Gray400,
                unfocusedPlaceholderColor = Gray400,
                disabledPlaceholderColor = Gray300,
                errorPlaceholderColor = Gray400
            )
        )

        // Supporting text
        if (supportingTextContent != null) {
            Text(
                text = supportingTextContent,
                style = MaterialTheme.typography.bodyMedium,
                color = when {
                    isError -> Red600
                    isSuccess -> Green600
                    else -> Gray600
                },
                modifier = Modifier.padding(
                    start = Dimen.PaddingSm,
                    top = Dimen.PaddingXs
                )
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
fun CustomTextFieldPreview() {
    NexemTheme {
        Column(
            modifier = Modifier.padding(Dimen.PaddingMd)
        ) {
            CustomTextField(
                value = "",
                onValueChange = { },
                type = TextFieldType.TEXT,
                label = "Text Field",
                placeholder = "Enter text...",
                helperText = "This is a normal text field",
                state = TextFieldState.NORMAL,
                modifier = Modifier.padding(bottom = Dimen.SpaceMd)
            )

            CustomTextField(
                value = "12345",
                onValueChange = { },
                type = TextFieldType.NUMBER,
                label = "Number Field",
                placeholder = "Enter numbers...",
                helperText = "Only digits allowed",
                state = TextFieldState.NORMAL,
                modifier = Modifier.padding(bottom = Dimen.SpaceMd)
            )

            CustomTextField(
                value = "password123",
                onValueChange = { },
                type = TextFieldType.PASSWORD,
                label = "Password Field",
                placeholder = "Enter password...",
                helperText = "Password with visibility toggle",
                state = TextFieldState.NORMAL,
                modifier = Modifier.padding(bottom = Dimen.SpaceMd)
            )

            CustomTextField(
                value = "Error text",
                onValueChange = { },
                label = "Error State",
                placeholder = "Enter text...",
                errorMessage = "This field has an error",
                state = TextFieldState.ERROR,
                modifier = Modifier.padding(bottom = Dimen.SpaceMd)
            )

            CustomTextField(
                value = "Success text",
                onValueChange = { },
                label = "Success State",
                placeholder = "Enter text...",
                successMessage = "This field is valid",
                state = TextFieldState.SUCCESS,
                modifier = Modifier.padding(bottom = Dimen.SpaceMd)
            )
        }
    }
}