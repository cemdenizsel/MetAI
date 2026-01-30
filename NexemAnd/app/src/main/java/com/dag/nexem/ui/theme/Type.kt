package com.dag.nexem.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

// Font families
val SansFont = FontFamily.Default
val MonoFont = FontFamily.Monospace

// Font sizes
val FontSizeXs = 12.sp
val FontSizeSm = 14.sp
val FontSizeBase = 16.sp
val FontSizeLg = 18.sp
val FontSizeXl = 20.sp
val FontSize2Xl = 24.sp

// Typography system following Material 3 guidelines
val Typography = Typography(
    // Display styles
    displayLarge = TextStyle(
        fontSize = FontSize2Xl,
        fontWeight = FontWeight.Medium,
        lineHeight = (FontSize2Xl.value * 1.5).sp
    ),
    displayMedium = TextStyle(
        fontSize = FontSizeXl,
        fontWeight = FontWeight.Medium,
        lineHeight = (FontSizeXl.value * 1.5).sp
    ),
    displaySmall = TextStyle(
        fontSize = FontSizeLg,
        fontWeight = FontWeight.Medium,
        lineHeight = (FontSizeLg.value * 1.5).sp
    ),
    
    // Headline styles
    headlineLarge = TextStyle(
        fontSize = FontSize2Xl,
        fontWeight = FontWeight.Medium,
        lineHeight = (FontSize2Xl.value * 1.5).sp
    ),
    headlineMedium = TextStyle(
        fontSize = FontSizeXl,
        fontWeight = FontWeight.Medium,
        lineHeight = (FontSizeXl.value * 1.5).sp
    ),
    headlineSmall = TextStyle(
        fontSize = FontSizeLg,
        fontWeight = FontWeight.Medium,
        lineHeight = (FontSizeLg.value * 1.5).sp
    ),
    
    // Title styles
    titleLarge = TextStyle(
        fontSize = FontSizeBase,
        fontWeight = FontWeight.Medium,
        lineHeight = (FontSizeBase.value * 1.5).sp
    ),
    titleMedium = TextStyle(
        fontSize = FontSizeSm,
        fontWeight = FontWeight.Medium,
        lineHeight = (FontSizeSm.value * 1.5).sp
    ),
    titleSmall = TextStyle(
        fontSize = FontSizeXs,
        fontWeight = FontWeight.Medium,
        lineHeight = (FontSizeXs.value * 1.5).sp
    ),
    
    // Body styles
    bodyLarge = TextStyle(
        fontSize = FontSizeBase,
        fontWeight = FontWeight.Normal,
        lineHeight = (FontSizeBase.value * 1.5).sp
    ),
    bodyMedium = TextStyle(
        fontSize = FontSizeSm,
        fontWeight = FontWeight.Normal,
        lineHeight = (FontSizeSm.value * 1.5).sp
    ),
    bodySmall = TextStyle(
        fontSize = FontSizeXs,
        fontWeight = FontWeight.Normal,
        lineHeight = (FontSizeXs.value * 1.5).sp
    ),
    
    // Label styles
    labelLarge = TextStyle(
        fontSize = FontSizeBase,
        fontWeight = FontWeight.Medium,
        lineHeight = (FontSizeBase.value * 1.5).sp
    ),
    labelMedium = TextStyle(
        fontSize = FontSizeSm,
        fontWeight = FontWeight.Medium,
        lineHeight = (FontSizeSm.value * 1.5).sp
    ),
    labelSmall = TextStyle(
        fontSize = FontSizeXs,
        fontWeight = FontWeight.Thin,
        lineHeight = (FontSizeXs.value * 1.5).sp,
        color = Gray500
    )
)