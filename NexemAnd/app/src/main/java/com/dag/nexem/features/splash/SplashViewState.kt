package com.dag.nexem.features.splash

import com.dag.nexem.base.architecture.BaseVS
import com.dag.nexem.base.navigation.Destination

sealed class SplashViewState: BaseVS {
    data class StartApp(val destination: Destination): SplashViewState()
    data object CloseApp: SplashViewState()
}