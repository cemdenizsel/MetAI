package com.dag.nexem.features.auth.intro

import androidx.annotation.StringRes
import androidx.compose.ui.graphics.vector.ImageVector

data class IntroContent(
    @StringRes val titleRes: Int,
    @StringRes val textRes: Int,
    val icon: ImageVector
)

data class IntroViewState(
    val contents: List<IntroContent> = emptyList(),
    val currentIndex: Int = 0
) {
    val totalPages: Int
        get() = contents.size
}
