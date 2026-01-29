package com.dag.nexem.base.navigation


import kotlinx.serialization.Serializable

sealed interface Destination {
    @Serializable
    data object Splash: Destination

    @Serializable
    data object HomeScreen: Destination

    @Serializable
    data object LoginScreen: Destination

    @Serializable
    data object RegisterScreen: Destination

    @Serializable
    data object PasswordActionScreen: Destination

    @Serializable
    data object PasswordResultScreen: Destination

    @Serializable
    data object Analytics: Destination

    @Serializable
    data object Settings: Destination

    companion object {
        val NAV_WITHOUT_BOTTOM_NAVBAR = listOf(Splash, LoginScreen, RegisterScreen,
            PasswordActionScreen, PasswordResultScreen)
    }

}
