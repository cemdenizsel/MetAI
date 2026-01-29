package com.dag.nexem

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ExitToApp
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.zIndex
import androidx.lifecycle.lifecycleScope
import com.dag.nexem.base.LocalScrollStateManager
import com.dag.nexem.base.ScrollStateManager
import com.dag.nexem.base.bottomnavigation.BottomNavManager
import com.dag.nexem.base.bottomnavigation.BottomNavigationBar
import com.dag.nexem.base.components.CustomAlertDialog
import com.dag.nexem.base.data.alertdialog.AlertDialogModel
import com.dag.nexem.base.helper.ActivityHolder
import com.dag.nexem.base.helper.AlertDialogManager
import com.dag.nexem.base.navigation.DefaultNavigationHost
import com.dag.nexem.base.navigation.DefaultNavigator
import com.dag.nexem.base.navigation.Destination
import com.dag.nexem.base.navigation.getDestinationTitle
import com.dag.nexem.ui.theme.Gray800
import com.dag.nexem.ui.theme.Gray900
import com.dag.nexem.ui.theme.NexemTheme
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import javax.inject.Inject

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    private val currentRoute = mutableStateOf<String?>(null)
    private val mainVM: MainVM by viewModels()

    @Inject
    lateinit var alertDialogManager: AlertDialogManager

    @Inject
    lateinit var scrollStateManager: ScrollStateManager

    @Inject
    lateinit var bottomNavManager: BottomNavManager

    @Inject
    lateinit var defaultNavigator: DefaultNavigator

    @Inject
    lateinit var activityHolder: ActivityHolder

    companion object {
        private const val TAG = "MainActivity"
    }


    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityHolder.setActivity(this)
        val showAlert = mutableStateOf(false)
        val alertDialogModel = mutableStateOf<AlertDialogModel?>(null)

        // Initialize alert dialog observer
        if (::alertDialogManager.isInitialized && lifecycleScope.isActive) {
            lifecycleScope.launch {
                alertDialogManager.alertFlow.collect { model ->
                    alertDialogModel.value = model
                    showAlert.value = true
                }
            }
        }

        setContent {
            val scrollState = scrollStateManager.scrollState.collectAsState()
            CompositionLocalProvider(
                LocalScrollStateManager provides scrollStateManager
            ) {
                NexemTheme {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(Gray900)
                    ) {
                        Surface(
                            modifier = Modifier.fillMaxSize(),
                            color = Color.Transparent,
                        ) {
                            Column(
                                verticalArrangement = Arrangement.SpaceBetween,
                                modifier = Modifier
                                    .fillMaxSize()
                                    .background(Gray900),
                            ) {
                                if (mainVM.isBottomNavActive(currentRoute.value)) {
                                    val title = currentRoute.value?.let { getDestinationTitle(it) } ?: ""
                                    TopAppBar(
                                        modifier = Modifier
                                            .fillMaxWidth(),
                                        colors = TopAppBarDefaults.topAppBarColors(
                                            containerColor = Gray800
                                        ),
                                        title = {
                                            Text(
                                                text = title,
                                                fontSize = 22.sp,
                                                fontWeight = FontWeight.Bold,
                                                color = Color.White
                                            )
                                        },
                                        actions = {
                                            IconButton(
                                                onClick = {
                                                    // Perform logout - clear all user data
                                                    lifecycleScope.launch {
                                                        // Clear auth data and navigate
                                                        mainVM.logout()
                                                        defaultNavigator.navigate(Destination.LoginScreen) {
                                                            launchSingleTop = true
                                                            popUpTo(0) { inclusive = true }
                                                        }
                                                    }
                                                }
                                            ) {
                                                Icon(
                                                    imageVector = Icons.AutoMirrored.Filled.ExitToApp,
                                                    contentDescription = "Sign Out",
                                                    tint = Color.White,
                                                    modifier = Modifier.size(24.dp)
                                                )
                                            }
                                        }
                                    )
                                }

                                DefaultNavigationHost(
                                    navigator = defaultNavigator,
                                    modifier = Modifier.weight(1f),
                                    startDestination = Destination.Splash
                                ) {
                                    currentRoute.value = it.destination.route
                                        ?.split(".")?.last()
                                }

                                if (mainVM.isBottomNavActive(currentRoute.value)) {
                                    BottomNavigationBar(
                                        currentRoute = currentRoute.value,
                                        isScrolled = scrollState.value.isScrolling,
                                        messageManager = bottomNavManager,
                                        onItemSelected = {
                                            lifecycleScope.launch {
                                                defaultNavigator.navigate(it) {
                                                    launchSingleTop = true
                                                    popUpTo(0) { inclusive = true }
                                                }
                                            }
                                        },
                                        onExpandClick = {
                                            scrollStateManager.toggle()
                                        }
                                    )
                                }
                            }
                        }
                        AnimatedVisibility(showAlert.value && alertDialogModel.value != null) {
                            Box(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .background(Color.Black.copy(alpha = 0.5f))
                                    .blur(16.dp)
                                    .zIndex(10f)
                            ) {
                                alertDialogModel.value?.let { model ->
                                    CustomAlertDialog(
                                        alertDialogModel = model,
                                        showAlert = showAlert,
                                        defaultNavigator = defaultNavigator
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}