package com.dag.nexem.base.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Build
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.tooling.preview.Preview
import com.dag.nexem.ui.theme.Dimen
import com.dag.nexem.ui.theme.NexemTheme
import com.dag.nexem.ui.theme.Purple600
import com.dag.nexem.ui.theme.White

@Composable
fun CustomButton(
    text: String,
    leftIcon: ImageVector? = null,
    rightIcon: ImageVector? = null,
    backgroundColor: Color = Purple600,
    onClick: () -> Unit
) {
    require(leftIcon == null || rightIcon == null) {
        "CustomButton cannot have both leftIcon and rightIcon set at the same time"
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(Dimen.RadiusMd))
            .background(backgroundColor)
            .clickable{
                onClick()
            },
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center,
    ) {
        leftIcon?.let {
            Icon(
                imageVector = it,
                contentDescription = "Left icon",
                tint = White
            )
        }
        Text(
            text,
            style = MaterialTheme.typography.displayMedium.copy(color = Color.White),
            modifier = Modifier.padding(Dimen.PaddingMd)
        )
        rightIcon?.let {
            Icon(
                imageVector = it,
                contentDescription = "Right icon",
                tint = White
            )
        }
    }
}

@Composable
@Preview
fun PreviewCustomButton() {
    NexemTheme {
        Surface {
            CustomButton(
                "Test Button"
            ) {}
        }
    }
}

@Composable
@Preview
fun PreviewCustomButtonLeftIcon() {
    NexemTheme {
        Surface {
            CustomButton(
                "Test Button",
                leftIcon = Icons.Filled.Build
            ) {}
        }
    }
}

@Composable
@Preview
fun PreviewCustomButtonRightIcon() {
    NexemTheme {
        Surface {
            CustomButton(
                "Test Button",
                rightIcon = Icons.Filled.Build
            ) {}
        }
    }
}