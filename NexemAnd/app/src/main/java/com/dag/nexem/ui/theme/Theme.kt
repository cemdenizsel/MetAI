package com.dag.nexem.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext

private val DarkColorScheme = darkColorScheme(
    primary = Indigo600,
    onPrimary = White,
    primaryContainer = Indigo900,
    onPrimaryContainer = Indigo100,
    
    secondary = Gray600,
    onSecondary = White,
    secondaryContainer = Gray800,
    onSecondaryContainer = Gray200,
    
    tertiary = Purple600,
    onTertiary = White,
    tertiaryContainer = Purple500,
    onTertiaryContainer = Purple100,
    
    background = Gray900,
    onBackground = Gray100,
    surface = Gray800,
    onSurface = Gray100,
    surfaceVariant = Gray700,
    onSurfaceVariant = Gray300,
    
    outline = Gray600,
    outlineVariant = Gray700,
    
    error = Red600,
    onError = White,
    errorContainer = Red500,
    onErrorContainer = Red100,
    
    surfaceTint = Indigo600,
    inverseSurface = Gray100,
    inverseOnSurface = Gray800,
    inversePrimary = Indigo400,
    
    // Custom semantic colors for the app
    surfaceBright = Gray700,
    surfaceDim = Gray900,
    surfaceContainer = Gray800,
    surfaceContainerHigh = Gray700,
    surfaceContainerHighest = Gray600,
    surfaceContainerLow = Gray800,
    surfaceContainerLowest = Gray900
)

private val LightColorScheme = lightColorScheme(
    primary = Indigo600,
    onPrimary = White,
    primaryContainer = Indigo100,
    onPrimaryContainer = Indigo900,
    
    secondary = Gray700,
    onSecondary = White,
    secondaryContainer = Gray100,
    onSecondaryContainer = Gray900,
    
    tertiary = Purple600,
    onTertiary = White,
    tertiaryContainer = Purple100,
    onTertiaryContainer = Purple600,
    
    background = White,
    onBackground = Gray900,
    surface = White,
    onSurface = Gray900,
    surfaceVariant = Gray100,
    onSurfaceVariant = Gray600,
    
    outline = Gray400,
    outlineVariant = Gray200,
    
    error = Red600,
    onError = White,
    errorContainer = Red100,
    onErrorContainer = Red600,
    
    surfaceTint = Indigo600,
    inverseSurface = Gray800,
    inverseOnSurface = Gray100,
    inversePrimary = Indigo200,
    
    // Custom semantic colors for the app
    surfaceBright = White,
    surfaceDim = Gray50,
    surfaceContainer = Gray100,
    surfaceContainerHigh = Gray50,
    surfaceContainerHighest = White,
    surfaceContainerLow = Gray100,
    surfaceContainerLowest = White
)

@Composable
fun NexemTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    // Dynamic color is available on Android 12+
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }

        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}