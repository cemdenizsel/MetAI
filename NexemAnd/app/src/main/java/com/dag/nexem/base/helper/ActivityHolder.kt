package com.dag.nexem.base.helper

import android.app.Activity
import java.lang.ref.WeakReference
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ActivityHolder @Inject constructor() {
    private var activityRef: WeakReference<Activity>? = null

    fun setActivity(activity: Activity) {
        activityRef = WeakReference(activity)
    }

    fun getActivity(): Activity? = activityRef?.get()

    fun clearActivity() {
        activityRef = null
    }
}