package com.application.ocr

import android.app.Activity
import android.content.pm.PackageManager
import android.view.View
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

/**
 * Created by Eray BULUT on 20.12.2023
 * eraybulutlar@gmail.com
 */


fun Activity.ensureCameraPermission(): Boolean {
    val cameraPermission = android.Manifest.permission.CAMERA
    val granted = ContextCompat.checkSelfPermission(
        this,
        cameraPermission
    ) == PackageManager.PERMISSION_GRANTED

    if (granted) return true

    ActivityCompat.requestPermissions(
        this,
        arrayOf(cameraPermission),
        101
    )
    return false
}

fun View.visible() {
    visibility = View.VISIBLE
}

fun View.gone() {
    visibility = View.GONE
}
