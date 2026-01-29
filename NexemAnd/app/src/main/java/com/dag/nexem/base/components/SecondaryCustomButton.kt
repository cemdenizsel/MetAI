package com.dag.nexem.base.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.ui.unit.dp
import com.dag.nexem.ui.theme.Dimen
import com.dag.nexem.ui.theme.NexemTheme
import com.dag.nexem.ui.theme.Purple600

@Composable
fun SecondaryCustomButton(
    text: String,
    leftIcon: ImageVector? = null,
    rightIcon: ImageVector? = null,
    onClick: () -> Unit = {}
) {
    require(leftIcon == null || rightIcon == null) {
        "SecondaryCustomButton cannot have both leftIcon and rightIcon set at the same time"
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(Dimen.RadiusMd))
            .clickable(onClick = onClick)
            .border(
                width = 2.dp,
                color = Purple600,
                shape = RoundedCornerShape(Dimen.RadiusMd)
            )
            .background(Color.White),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center,
    ) {
        leftIcon?.let {
            Icon(
                imageVector = it,
                contentDescription = "Left icon",
                tint = Purple600
            )
        }
        Text(
            text,
            style = MaterialTheme.typography.displayMedium.copy(color = Purple600),
            modifier = Modifier.padding(Dimen.PaddingMd)
        )
        rightIcon?.let {
            Icon(
                imageVector = it,
                contentDescription = "Right icon",
                tint = Purple600
            )
        }
    }
}

@Composable
@Preview
fun PreviewSecondaryCustomButton() {
    NexemTheme {
        Surface {
            SecondaryCustomButton(
                "Test Button"
            )
        }
    }
}

@Composable
@Preview
fun PreviewSecondaryCustomButtonLeftIcon() {
    NexemTheme {
        Surface {
            SecondaryCustomButton(
                "Test Button",
                leftIcon = Icons.Filled.Build
            )
        }
    }
}

@Composable
@Preview
fun PreviewSecondaryCustomButtonRightIcon() {
    NexemTheme {
        Surface {
            SecondaryCustomButton(
                "Test Button",
                rightIcon = Icons.Filled.Build
            )
        }
    }
}