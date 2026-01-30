package com.dag.nexem.features.auth.di

import com.dag.nexem.features.auth.data.remote.AuthApiService
import com.dag.nexem.features.auth.data.repository.AuthRepositoryImpl
import com.dag.nexem.features.auth.domain.repository.AuthRepository
import com.dag.nexem.network.AuthenticatedRetrofit
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import retrofit2.Retrofit
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AuthModule {

    @Provides
    @Singleton
    fun provideAuthApiService(
        @AuthenticatedRetrofit retrofit: Retrofit
    ): AuthApiService {
        return retrofit.create(AuthApiService::class.java)
    }

    @Provides
    @Singleton
    fun provideAuthRepository(
        authRepositoryImpl: AuthRepositoryImpl
    ): AuthRepository {
        return authRepositoryImpl
    }
}
