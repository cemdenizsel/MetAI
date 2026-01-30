package com.dag.nexem.features.splash

import androidx.lifecycle.viewModelScope
import com.dag.nexem.base.architecture.BaseVM
import com.dag.nexem.base.helper.ActivityHolder
import com.dag.nexem.base.helper.AlertDialogManager
import com.dag.nexem.base.navigation.Destination
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import javax.inject.Inject


@HiltViewModel
class SplashViewModel @Inject constructor(
    private val alertDialogManager: AlertDialogManager,
    private val activityHolder: ActivityHolder,
): BaseVM<SplashViewState>(){

    init {
        startApp()
    }

    companion object {
        private const val METAMASK_PACKAGE_NAME = "io.metamask"
        private const val SPLASH_DELAY = 3000L // 3 seconds delay
    }

    fun startApp() {
        viewModelScope.launch {
            delay(SPLASH_DELAY)
            // Always go to login screen to ensure fresh authentication
            _viewState.value = SplashViewState.StartApp(Destination.LoginScreen)
        }
    }

}