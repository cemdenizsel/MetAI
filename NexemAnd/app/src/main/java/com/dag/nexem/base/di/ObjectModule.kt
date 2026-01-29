package com.dag.nexem.base.di

import com.dag.nexem.base.ScrollStateManager
import com.dag.nexem.base.helper.AlertDialogManager
import com.dag.nexem.base.navigation.DefaultNavigator
import com.dag.nexem.base.navigation.Destination
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton


@Module
@InstallIn(SingletonComponent::class)
class ObjectModules {

    @Provides
    @Singleton
    fun provideDefaultNavigator(): DefaultNavigator {
        return DefaultNavigator(startDestination = Destination.Splash)
    }

    @Provides
    @Singleton
    fun provideAlertDialogManager(): AlertDialogManager {
        return AlertDialogManager()
    }

    @Provides
    @Singleton
    fun provideScrollStateManager(): ScrollStateManager {
        return ScrollStateManager()
    }

}