package com.dag.nexem.base.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Build
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.tooling.preview.Preview
import com.dag.nexem.ui.theme.Purple600
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.ui.Alignment
import androidx.compose.ui.unit.dp
import com.dag.nexem.ui.theme.Black
import com.dag.nexem.ui.theme.Dimen
import com.dag.nexem.ui.theme.White


@Composable
fun CustomCard(
    modifier: Modifier = Modifier,
    tintColor: Color = White,
    backgroundColor: Color = Purple600,
    subtitle: String? = null,
    textColor: Color = Black,
    icon: ImageVector,
    title: String
) {
    Column(
        modifier = modifier.padding(Dimen.PaddingMd),
        verticalArrangement = Arrangement.spacedBy(Dimen.SpaceMd),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        CardHeader(icon = icon, tintColor = tintColor, backgroundColor = backgroundColor)
        Text(title, style = MaterialTheme.typography.titleLarge.copy(textColor))
        subtitle?.let {
            Text(it, style = MaterialTheme.typography.bodyMedium.copy(textColor))
        }
    }
}

@Composable
fun CardHeader(
    modifier: Modifier = Modifier,
    icon: ImageVector,
    tintColor: Color,
    backgroundColor: Color
) {
    Box(
        modifier = modifier
            .clip(CircleShape)
            .background(backgroundColor),
        contentAlignment = Alignment.Center
    ) {
        Icon(
            imageVector = icon,
            contentDescription = "Icon",
            modifier = Modifier.padding(16.dp),
            tint = tintColor
        )
    }
}

@Composable
@Preview
fun CardPreview() {
    Surface {
        CustomCard(icon = Icons.Default.Build, title = "Test card", subtitle = "Test subtitle")
    }
}