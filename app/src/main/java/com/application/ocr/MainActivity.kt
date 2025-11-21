package com.application.ocr

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.application.ocr.databinding.ActivityMainBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.util.ArrayDeque
import java.util.LinkedHashSet
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private var scannedText: String? = null
    private lateinit var cameraExecutor: ExecutorService
    private var analysisUseCase: ImageAnalysis? = null
    private var processCameraProvider: ProcessCameraProvider? = null
    @Volatile
    private var isProcessingFrame = false
    private val licensePlatePattern = Regex("\\b[A-Z0-9]{6,7}\\b")
    private val detectedPlates = LinkedHashSet<String>()
    private val plateDetectionHistory = ArrayDeque<Set<String>>()
    private val historyLimit = 30
    private val minPlateOccurrences = 3
    private val maxLineCountForPlate = 4

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)
        cameraExecutor = Executors.newSingleThreadExecutor()

        initializeListeners()
        if (ensureCameraPermission()) {
            startCameraPreview()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    private fun initializeListeners() {
        with(binding) {
            btnCopy.onVibrationClick {
                if (scannedText.isNullOrEmpty()) {
                    showToast("Please scan an image")
                    return@onVibrationClick
                }
                showToast("Copied to clipboard")
                copyToClipboard(scannedText!!)
            }

            btnSend.onVibrationClick {
                if (scannedText.isNullOrEmpty()) {
                    showToast("Please scan an image")
                    return@onVibrationClick
                }
                showToast("Sending...")
                shareText(scannedText!!)
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 101) {
            val granted = grantResults.all { it == android.content.pm.PackageManager.PERMISSION_GRANTED }
            if (granted) {
                startCameraPreview()
            } else {
                showToast("Camera permission is required for live preview")
            }
        }
    }

    private fun startCameraPreview() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            processCameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = processCameraProvider ?: return
        val preview = Preview.Builder()
            .build()
            .also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

        analysisUseCase = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    processImage(imageProxy)
                }
            }

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                analysisUseCase
            )
        } catch (exc: Exception) {
            Log.e("CameraX", "Binding failed", exc)
        }
    }

    private fun processImage(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image ?: run {
            imageProxy.close()
            return
        }

        if (isProcessingFrame) {
            imageProxy.close()
            return
        }

        isProcessingFrame = true
        val inputImage =
            InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)

        recognizer.process(inputImage)
            .addOnSuccessListener { visionText ->
                scannedText = visionText.text
                runOnUiThread {
                    binding.resultTextView.text =
                        if (visionText.text.isBlank()) "No text detected" else visionText.text
                    updateDetectedPlates(visionText.text)
                }
            }
            .addOnFailureListener { exception ->
                Log.e("CameraX", "Text recognition failed", exception)
            }
            .addOnCompleteListener {
                imageProxy.close()
                isProcessingFrame = false
            }
    }

    private fun updateDetectedPlates(rawText: String) {
        val lineCount = rawText.lines().count { it.isNotBlank() }
        val uppercaseText = rawText.uppercase()
        val matches = licensePlatePattern.findAll(uppercaseText).map { it.value }.toSet()
        recordPlateHistory(matches)

        if (lineCount > maxLineCountForPlate || matches.isEmpty()) {
            return
        }

        var newPlateAdded = false
        matches.forEach { candidate ->
            val occurrences = plateDetectionHistory.count { it.contains(candidate) }
            if (occurrences >= minPlateOccurrences && detectedPlates.add(candidate)) {
                newPlateAdded = true
            }
        }

        if (newPlateAdded) {
            refreshPlateListText()
        }
    }

    private fun recordPlateHistory(matches: Set<String>) {
        plateDetectionHistory.addLast(matches)
        if (plateDetectionHistory.size > historyLimit) {
            plateDetectionHistory.removeFirst()
        }
    }

    private fun refreshPlateListText() {
        binding.plateListTextView.text =
            if (detectedPlates.isEmpty()) {
                getString(R.string.no_license_plate_detected)
            } else {
                detectedPlates.joinToString(separator = "\n")
            }
    }
}
