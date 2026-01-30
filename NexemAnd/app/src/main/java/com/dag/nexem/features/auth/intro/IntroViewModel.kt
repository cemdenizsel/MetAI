package com.dag.nexem.features.auth.intro

import androidx.compose.ui.graphics.vector.ImageVector
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

class IntroViewModel : ViewModel() {

    private val _viewState = MutableStateFlow(IntroViewState())
    val viewState: StateFlow<IntroViewState> = _viewState.asStateFlow()

    fun initializeContent(titles: List<Int>, text: List<Int>, icons: List<ImageVector>) {
        val contents = titles
            .zip(text)
            .mapIndexed { index, (titleRes, textRes) ->
                IntroContent(
                    titleRes = titleRes,
                    textRes = textRes,
                    icon = icons[index]
                )
            }
        _viewState.update { it.copy(contents = contents) }
    }

    fun updateCurrentIndex(index: Int) {
        _viewState.update { it.copy(currentIndex = index) }
    }

    fun canSwipeLeft(): Boolean {
        return _viewState.value.currentIndex > 0
    }

    fun canSwipeRight(): Boolean {
        return _viewState.value.currentIndex < _viewState.value.contents.size - 1
    }

    fun isLastPage(): Boolean {
        return _viewState.value.currentIndex == _viewState.value.contents.size - 1
    }

    fun onNextClick() {
        if (canSwipeRight()) {
            _viewState.update { it.copy(currentIndex = it.currentIndex + 1) }
        }
    }

    fun onSkip() {
        // TODO: Navigate to next screen
    }

    fun onGetStarted() {
        // TODO: Navigate to next screen
    }
}
