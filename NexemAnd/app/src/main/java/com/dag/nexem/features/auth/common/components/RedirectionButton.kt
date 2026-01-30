package com.dag.nexem.features.auth.common.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.dag.nexem.ui.theme.Dimen
import com.dag.nexem.ui.theme.NexemTheme
import com.dag.nexem.ui.theme.White

@Composable
fun RedirectionButton(
    text: String,
    textColor: Color = White,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier.height(Dimen.ButtonHeightMd),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .weight(1f)
                .height(1.dp)
                .background(Color.Gray)
        )

        Spacer(modifier = Modifier.size(Dimen.SpaceMd))

        Text(
            text = text,
            style = MaterialTheme.typography.displaySmall.copy(textColor),
            modifier = Modifier.clickable { onClick() }
        )
        Spacer(modifier = Modifier.size(Dimen.SpaceMd))
        Box(
            modifier = Modifier
                .weight(1f)
                .height(1.dp)
                .background(Color.Gray)
        )
    }
}

@Composable
@Preview
fun RedirectionButtonPreview() {
    NexemTheme {
        RedirectionButton("Sign In") {

        }
    }
}