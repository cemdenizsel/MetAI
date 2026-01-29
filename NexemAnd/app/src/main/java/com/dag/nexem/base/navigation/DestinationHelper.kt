package com.dag.nexem.base.navigation

import androidx.compose.runtime.Composable
import androidx.compose.ui.res.stringResource
import com.dag.nexem.R


@Composable
fun getDestinationTitle(destination: String): String{
    return when(destination) {
        Destination.HomeScreen.toString() -> {
            stringResource(R.string.app_name)
        }
        else -> {
            stringResource(R.string.app_name)
        }
    }
}