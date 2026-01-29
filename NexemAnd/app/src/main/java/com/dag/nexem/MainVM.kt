package com.dag.nexem


import androidx.lifecycle.viewModelScope
import com.dag.nexem.base.architecture.BaseVM
import com.dag.nexem.base.navigation.DefaultNavigator
import com.dag.nexem.base.navigation.Destination
import com.dag.nexem.network.TokenManager
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.launch
import javax.inject.Inject


@HiltViewModel
class MainVM @Inject constructor(
    private var defaultNavigator: DefaultNavigator,
    private val tokenManager: TokenManager
): BaseVM<MainVS>() {

    fun navigate(destination: Destination){
        viewModelScope.launch {
            defaultNavigator.navigate(destination)
        }
    }

    fun isBottomNavActive(currentRoute:String?): Boolean {
        return currentRoute?.let {
            return Destination.NAV_WITHOUT_BOTTOM_NAVBAR
                .map { it.toString() }.contains(currentRoute).not()
        } ?: false
    }

    fun logout() {
        viewModelScope.launch {
            try {
                tokenManager.clearToken()
            } catch (e: Exception) {
                android.util.Log.e("MainVM", "Error during logout", e)
            }
        }
    }

}