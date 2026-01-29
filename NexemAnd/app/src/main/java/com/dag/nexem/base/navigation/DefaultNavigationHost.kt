package com.dag.nexem.base.navigation

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.navigation.NavBackStackEntry
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.rememberNavController
import com.dag.nexem.base.extensions.ObserveAsEvents
import com.dag.nexem.features.auth.login.features.LoginScreen
import com.dag.nexem.features.auth.password.action.PasswordActionScreen
import com.dag.nexem.features.auth.password.result.PasswordResultScreen
import com.dag.nexem.features.auth.register.features.RegisterScreen
import com.dag.nexem.features.splash.SplashScreen

@Composable
fun DefaultNavigationHost(
    modifier: Modifier = Modifier,
    startDestination: Destination = Destination.Splash,
    navigator: DefaultNavigator,
    navBackStackEntryState: (NavBackStackEntry) -> Unit,
) {
    val navController = rememberNavController()
    ObserveAsEvents(flow = navigator.navigationActions) { action ->
        when (action) {
            is NavigationAction.Navigate -> navController.navigate(action.destination){
                action.navOptions(this)
            }
            NavigationAction.NavigateUp -> navController.navigateUp()
        }
    }
    ObserveAsEvents(flow = navController.currentBackStackEntryFlow){
        navBackStackEntryState(it)
    }
    Box(modifier = modifier.fillMaxSize()) {
        NavHost(
            navController = navController,
            modifier = Modifier.fillMaxSize(),
            startDestination = startDestination
        ) {
            splashComposable<Destination.Splash> {
                SplashScreen(
                    navController = navController
                )
            }

            composableWithAnimations<Destination.LoginScreen> {
                LoginScreen(
                    navController
                )
            }

            composableWithAnimations<Destination.RegisterScreen> {
                RegisterScreen(
                    navController
                )
            }

            composableWithAnimations<Destination.PasswordActionScreen> {
                PasswordActionScreen(navController)
            }

            composableWithAnimations<Destination.PasswordResultScreen> {
                PasswordResultScreen(navController)
            }

        }
    }
}