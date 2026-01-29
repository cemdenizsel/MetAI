package com.dag.nexem.base.bottomnavigation

import androidx.annotation.DrawableRes
import com.dag.nexem.R
import com.dag.nexem.base.navigation.Destination

enum class BottomNavIcon(
    @DrawableRes var icon: Int,
    var destination: Destination
) {
    Home(R.drawable.outline_home, Destination.HomeScreen),
    Analytics(R.drawable.outline_analytics, Destination.Analytics),
    Settings(R.drawable.outline_settings, Destination.Settings)
}