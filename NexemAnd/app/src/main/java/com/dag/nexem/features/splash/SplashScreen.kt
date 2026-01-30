package com.dag.nexem.features.splash

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.scale
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.navigation.NavHostController
import com.dag.nexem.R
import com.dag.nexem.ui.theme.White

@Composable
fun SplashScreen(
    navController: NavHostController,
    viewModel: SplashViewModel = hiltViewModel()
) {
    // Animation properties
    val infiniteTransition = rememberInfiniteTransition(label = "splash")

    // Collect state properly
    val viewState by viewModel.viewState.collectAsState()

    // Handle navigation effects
    LaunchedEffect(viewState) {
        when (viewState) {
            is SplashViewState.StartApp -> {
                val destination = (viewState as SplashViewState.StartApp).destination
                try {
                    navController.navigate(destination) {
                        launchSingleTop = true
                        popUpTo(0) { inclusive = true }
                    }
                } catch (e: Exception) {
                    // Handle navigation error if needed
                }
            }

            SplashViewState.CloseApp -> {
            }

            else -> { /* do nothing */
            }
        }
    }

    // Scale animation
    val scale by infiniteTransition.animateFloat(
        initialValue = 0.8f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(2000),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scale"
    )

    // Alpha animation
    val alpha by animateFloatAsState(
        targetValue = 1f,
        animationSpec = tween(1000),
        label = "alpha",
    )

    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .fillMaxSize()
            .background(White)
    ) {
        Image(
            painter = painterResource(R.drawable.ic_launcher_background),
            contentDescription = "App Logo",
            contentScale = ContentScale.Fit,
            modifier = Modifier
                .size(600.dp)
                .scale(scale)
                .alpha(alpha)
        )
    }
}