package com.dag.nexem.base.extensions

import androidx.navigation.NavController
import com.dag.nexem.base.navigation.Destination

fun NavController.startAsTopComposable(destination: Destination){
    this.navigate(destination) {
        launchSingleTop = true
        popUpTo(0) { inclusive = true }
    }
}