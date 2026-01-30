package com.dag.nexem.features.auth.intro

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.RecordVoiceOver
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import com.dag.nexem.R
import com.dag.nexem.base.components.CustomButton
import com.dag.nexem.base.components.CustomCard
import com.dag.nexem.base.components.SecondaryCustomButton
import com.dag.nexem.ui.theme.Dimen
import com.dag.nexem.ui.theme.Gray200
import com.dag.nexem.ui.theme.NexemTheme
import com.dag.nexem.ui.theme.Purple100
import com.dag.nexem.ui.theme.Purple600
import com.dag.nexem.ui.theme.White

@Composable
fun IntroScreen(
    viewModel: IntroViewModel = hiltViewModel()
) {
    val uiState by viewModel.viewState.collectAsState()

    // Load all strings first
    val titles = remember {
        listOf(
            R.string.intro_screen_title_state1,
            R.string.intro_screen_title_state2,
            R.string.intro_screen_title_state3
        )
    }
    val texts = remember {
        listOf(
            R.string.intro_screen_text_state1,
            R.string.intro_screen_text_state2,
            R.string.intro_screen_text_state3
        )
    }
    val icons = remember {
        listOf(Icons.Default.Psychology, Icons.Default.RecordVoiceOver, Icons.Default.Lock)
    }

    // Initialize content
    LaunchedEffect(Unit) {
        viewModel.initializeContent(titles,texts, icons)
    }

    // Create pager state
    val pagerState = rememberPagerState(pageCount = { uiState.totalPages })

    // Sync pager state with viewModel (when user swipes)
    LaunchedEffect(pagerState.currentPage) {
        viewModel.updateCurrentIndex(pagerState.currentPage)
    }

    // Sync viewModel state with pager (when viewModel changes)
    LaunchedEffect(uiState.currentIndex) {
        if (pagerState.currentPage != uiState.currentIndex) {
            pagerState.animateScrollToPage(uiState.currentIndex)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(White)
            .padding(Dimen.PaddingLg),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Progress indicators
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = Dimen.PaddingMd),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            repeat(uiState.totalPages) { index ->
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .height(4.dp)
                        .background(
                            color = if (index == uiState.currentIndex) Purple600 else Gray200,
                            shape = RoundedCornerShape(2.dp)
                        )
                )
                if (index < uiState.totalPages - 1) {
                    Spacer(modifier = Modifier.width(Dimen.SpaceSm))
                }
            }
        }

        Spacer(modifier = Modifier.weight(1f))

        // Content pager
        if (uiState.totalPages > 0) {
            HorizontalPager(
                state = pagerState,
                modifier = Modifier.fillMaxWidth(),
                userScrollEnabled = true
            ) { page ->
                val pageContent = uiState.contents.getOrNull(page)
                if (pageContent != null) {
                    CustomCard(
                        icon = pageContent.icon,
                        title = stringResource(pageContent.titleRes),
                        subtitle = stringResource(pageContent.textRes),
                        backgroundColor = Purple100,
                        tintColor = Purple600
                    )
                }
            }
        }

        Spacer(modifier = Modifier.weight(1f))

        // Buttons
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(Dimen.SpaceMd)
        ) {
            if (viewModel.isLastPage()) {
                CustomButton(
                    text = stringResource(R.string.intro_screen_finish_button)
                ) {}
            } else {
                CustomButton(
                    text = stringResource(R.string.intro_screen_next_button)
                ) {
                    viewModel.onNextClick()
                }
            }

            if (!viewModel.isLastPage()) {
                SecondaryCustomButton(
                    text = stringResource(R.string.intro_screen_skip_button)
                )
            }
        }
    }
}

@Composable
@Preview
fun IntroScreenPreview() {
    NexemTheme {
        IntroScreen()
    }
}
