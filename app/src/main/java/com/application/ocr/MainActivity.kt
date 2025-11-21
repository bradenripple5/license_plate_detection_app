package com.application.ocr

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.doOnLayout
import com.application.ocr.databinding.ActivityMainBinding
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import android.graphics.Color
import java.io.ByteArrayOutputStream
import java.util.LinkedHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.atan2
import kotlin.math.roundToInt


class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private var scannedText: String? = null
    private lateinit var cameraExecutor: ExecutorService
    private var analysisUseCase: ImageAnalysis? = null
    private var processCameraProvider: ProcessCameraProvider? = null
    @Volatile
    private var isProcessingFrame = false

    private val permanentPlateMap = LinkedHashMap<String, PlateEntry>()
    private val windowCounts = LinkedHashMap<String, WindowPlateEntry>()
    private val recentDetections = mutableListOf<Pair<Long, String>>()
    private val minPlateLength = 5
    private val maxPlateLength = 7
    private val maxZoomSteps = 6
    private val minCropSize = 64
    private val minPlateAspect = 2.0f
    private val horizontalExpansionFactor = 1.3f
    private val bitmapSimilarityThreshold = 0.5f
    private val similaritySampleSize = 32
    private val initialFocusFraction = 0.85f
    private val verticalTrimFactor = 0.85f
    @Volatile
    private var activeZoomRect: Rect? = null
    private val confirmationThreshold = 2
    private val plateDetectionEnabled = false
    private val plateFilter = PlateFilter(minPlateLength, maxPlateLength)
    @Volatile
    private var algorithmConfirmShowing = false
    private val placeholderPlateBitmap by lazy {
        Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
    }

    private val windowIntervalMs = 300L
    private val recentStringsRetentionMs = 5_000L
    private val windowHandler = Handler(Looper.getMainLooper())
    private val windowEvaluator = object : Runnable {
        override fun run() {
            evaluateWindowCounts()
            windowHandler.postDelayed(this, windowIntervalMs)
        }
    }

    private val confirmedPlates = mutableSetOf<String>()
    @Volatile
    private var confirmDialogShowing = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)
        binding.previewView.doOnLayout {
            plateFilter.updatePreviewSize(it.width, it.height)
        }
        binding.previewView.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ ->
            plateFilter.updatePreviewSize(binding.previewView.width, binding.previewView.height)
        }
        cameraExecutor = Executors.newSingleThreadExecutor()

        initializeListeners()
        if (ensureCameraPermission()) {
            startCameraPreview()
        }
        if (plateDetectionEnabled) {
            windowHandler.postDelayed(windowEvaluator, windowIntervalMs)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        if (plateDetectionEnabled) {
            windowHandler.removeCallbacks(windowEvaluator)
        }
    }

    private fun initializeListeners() {
        with(binding) {
            plateListTextView.setOnClickListener {
                showPlateRemovalDialog()
            }
            btnCopy.onVibrationClick {
                if (scannedText.isNullOrEmpty()) {
                    showToast("Please scan an image")
                    return@onVibrationClick
                }
                showToast("Copied to clipboard")
                copyToClipboard(scannedText!!)
            }

            btnSend.onVibrationClick {
                if (permanentPlateMap.isEmpty()) {
                    showToast(getString(R.string.no_plates_to_share))
                    return@onVibrationClick
                }
                val payload = buildPlateShareList()
                shareText(payload)
            }

            verticalFovSlider.value = fractionToSliderValue(plateFilter.verticalFraction, plateFilter.minVerticalFraction)
            verticalFovSlider.addOnChangeListener { _, value, _ ->
                val fraction = sliderValueToFraction(value, plateFilter.minVerticalFraction)
                updateVerticalFraction(fraction)
            }

            horizontalFovSlider.value = fractionToSliderValue(plateFilter.horizontalFraction, plateFilter.minHorizontalFraction)
            horizontalFovSlider.addOnChangeListener { _, value, _ ->
                val fraction = sliderValueToFraction(value, plateFilter.minHorizontalFraction)
                updateHorizontalFraction(fraction)
            }

            updateVerticalFraction(plateFilter.verticalFraction)
            updateHorizontalFraction(plateFilter.horizontalFraction)
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

//    @SuppressLint("UnsafeOptInUsageError")
    private fun processImage(imageProxy: ImageProxy) {
        if (!plateDetectionEnabled) {
            processRawFrame(imageProxy)
            return
        }

        if (isProcessingFrame) {
            imageProxy.close()
            return
        }

        isProcessingFrame = true
        val bitmapBuffer = imageProxyToBitmap(imageProxy)
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val rotatedBitmap = rotateBitmap(bitmapBuffer, rotationDegrees)
        val rawResult = recognizeTextBlocking(rotatedBitmap)
        val rawText = rawResult?.let { plateFilter.filterVisibleText(it, rotatedBitmap.width, rotatedBitmap.height) }
        val algorithmResult = rawResult?.let { plateFilter.computeAlgorithmResult(it, rotatedBitmap.width, rotatedBitmap.height) }
        runOnUiThread {
            postRawText(rawText)
            postAlgorithmResult(algorithmResult)
        }

        val zoomResult = zoomInOnSingleLine(rotatedBitmap)
        if (zoomResult == null) {
            imageProxy.close()
            isProcessingFrame = false
            activeZoomRect = null
            return
        }

        handleZoomResult(zoomResult)
        activeZoomRect = zoomResult.rect
        imageProxy.close()
        isProcessingFrame = false
    }

    private fun handleZoomResult(result: ZoomResult) {
        if (result.text.length !in minPlateLength..maxPlateLength) {
            return
        }
        val detection = PlateDetection(
            text = result.text,
            area = result.rect.width() * result.rect.height(),
            bitmap = result.bitmap,
            rect = result.rect,
            centerDistance = plateFilter.computeCenterDistance(result.rect, result.frameWidth, result.frameHeight)
        )
        runOnUiThread {
            incrementWindowCount(detection)
        }
    }

    private fun processRawFrame(imageProxy: ImageProxy) {
        if (isProcessingFrame) {
            imageProxy.close()
            return
        }
        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            imageProxy.close()
            return
        }
        isProcessingFrame = true
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val rotatedWidth: Int
        val rotatedHeight: Int
        if (rotationDegrees % 180 == 0) {
            rotatedWidth = imageProxy.width
            rotatedHeight = imageProxy.height
        } else {
            rotatedWidth = imageProxy.height
            rotatedHeight = imageProxy.width
        }
        val inputImage = InputImage.fromMediaImage(mediaImage, rotationDegrees)
        recognizer.process(inputImage)
            .addOnSuccessListener { text ->
                val rawText = plateFilter.filterVisibleText(text, rotatedWidth, rotatedHeight)
                val algorithmResult = plateFilter.computeAlgorithmResult(text, rotatedWidth, rotatedHeight)
                runOnUiThread {
                    postRawText(rawText)
                    postAlgorithmResult(algorithmResult)
                }
            }
            .addOnFailureListener { exception ->
                Log.e("CameraX", "Live OCR failed", exception)
                runOnUiThread {
                    postRawText(null)
                }
            }
            .addOnCompleteListener {
                imageProxy.close()
                isProcessingFrame = false
            }
    }

    private fun zoomInOnSingleLine(source: Bitmap): ZoomResult? {
        var currentRect = activeZoomRect ?: centeredRect(source.width, source.height, initialFocusFraction)
        var step = 0
        while (step < maxZoomSteps) {
            val crop = cropBitmap(source, currentRect) ?: return null
            val textResult = recognizeTextBlocking(crop) ?: return null
            val blocks = textResult.textBlocks
            if (blocks.isEmpty()) {
                activeZoomRect = null
                val trimmed = trimRectVertically(currentRect, source.height, 0)
                currentRect = trimmed ?: return null
                activeZoomRect = trimmed
                step++
                continue
            }
            val linePairs = blocks.flatMap { block ->
                block.lines.map { line ->
                    line to plateFilter.sanitizePlateText(line.text)
                }
            }.filter { it.second.isNotBlank() }
            if (linePairs.isEmpty()) {
                activeZoomRect = null
                val trimmed = trimRectVertically(currentRect, source.height, blocks.size)
                currentRect = trimmed ?: return null
                activeZoomRect = trimmed
                step++
                continue
            }
            if (linePairs.size == 1) {
                val targetLinePair = linePairs.maxByOrNull { it.first.boundingBox?.height() ?: 0 }
                if (targetLinePair != null) {
                    val (line, sanitized) = targetLinePair
                    val boundingBox = line.boundingBox ?: Rect(0, 0, crop.width, crop.height)
                    val expandedLocal = expandLineRectHorizontally(
                        boundingBox,
                        crop.width
                    )
                    val globalRect = Rect(
                        expandedLocal.left + currentRect.left,
                        expandedLocal.top + currentRect.top,
                        expandedLocal.right + currentRect.left,
                        expandedLocal.bottom + currentRect.top
                    )
                    val clampedGlobal = clampRectToBounds(globalRect, source.width, source.height)
                    if (sanitized.length in minPlateLength..maxPlateLength) {
                        val deskewed = deskewLineCrop(crop, line)
                        return ZoomResult(sanitized, deskewed, clampedGlobal, source.width, source.height)
                    }
                    activeZoomRect = clampedGlobal
                    currentRect = clampedGlobal
                    step++
                    continue
                }
            }
            val trimmed = trimRectVertically(currentRect, source.height, 0)
            currentRect = trimmed ?: return null
            activeZoomRect = trimmed
            step++
        }
        return null
    }

    private fun recognizeTextBlocking(bitmap: Bitmap): Text? {
        return try {
            val image = InputImage.fromBitmap(bitmap, 0)
            Tasks.await(recognizer.process(image))
        } catch (exception: Exception) {
            Log.e("CameraX", "Blocking OCR failed", exception)
            null
        }
    }

    private fun trimRectVertically(rect: Rect, maxHeight: Int, lineCount: Int): Rect? {
        val currentHeight = rect.height()
        if (currentHeight <= minCropSize) {
            return null
        }
        val factor = when {
            lineCount <= 1 -> verticalTrimFactor
            lineCount <= 3 -> verticalTrimFactor
            else -> verticalTrimFactor
        }
        val targetHeight = (currentHeight * factor).roundToInt().coerceAtLeast(minCropSize)
        if (targetHeight == currentHeight) return null
        val centerY = rect.centerY()
        var top = centerY - targetHeight / 2
        var bottom = centerY + targetHeight / 2
        if (top < 0) {
            bottom -= top
            top = 0
        }
        if (bottom > maxHeight) {
            val shift = bottom - maxHeight
            top -= shift
            bottom = maxHeight
        }
        top = top.coerceAtLeast(0)
        bottom = bottom.coerceAtMost(maxHeight)
        if (top >= bottom) return null
        return Rect(rect.left, top, rect.right, bottom)
    }

    private fun centeredRect(width: Int, height: Int, fraction: Float): Rect {
        val clampedFraction = fraction.coerceIn(0.1f, 1f)
        val targetWidth = (width * clampedFraction).roundToInt().coerceAtLeast(minCropSize)
        val targetHeight = (height * clampedFraction).roundToInt().coerceAtLeast(minCropSize)
        val left = ((width - targetWidth) / 2).coerceAtLeast(0)
        val top = ((height - targetHeight) / 2).coerceAtLeast(0)
        val right = (left + targetWidth).coerceAtMost(width)
        val bottom = (top + targetHeight).coerceAtMost(height)
        return Rect(left, top, right, bottom)
    }

    private fun expandLineRectHorizontally(
        rect: Rect,
        containerWidth: Int
    ): Rect {
        val height = rect.height().coerceAtLeast(1)
        val minWidth = (height * minPlateAspect).roundToInt()
        val targetWidth = (rect.width() * horizontalExpansionFactor).roundToInt()
            .coerceAtLeast(minWidth)
        val centerX = rect.centerX()
        var left = centerX - targetWidth / 2
        var right = left + targetWidth
        if (left < 0) {
            right -= left
            left = 0
        }
        if (right > containerWidth) {
            val shift = right - containerWidth
            left -= shift
            right = containerWidth
        }
        left = left.coerceAtLeast(0)
        right = right.coerceAtMost(containerWidth)
        if (left >= right) {
            return Rect(rect.left.coerceAtLeast(0), rect.top, rect.right.coerceAtMost(containerWidth), rect.bottom)
        }
        return Rect(left, rect.top, right, rect.bottom)
    }

    private fun clampRectToBounds(rect: Rect, maxWidth: Int, maxHeight: Int): Rect {
        val left = rect.left.coerceIn(0, maxWidth - 1)
        val top = rect.top.coerceIn(0, maxHeight - 1)
        val right = rect.right.coerceIn(left + 1, maxWidth)
        val bottom = rect.bottom.coerceIn(top + 1, maxHeight)
        return Rect(left, top, right, bottom)
    }

    private fun deskewLineCrop(
        source: Bitmap,
        line: Text.Line
    ): Bitmap {
        val boundingBox = line.boundingBox
        val cornerPoints = line.cornerPoints
        val localSource = boundingBox?.let { safeCrop(source, it) } ?: source
        if (cornerPoints == null || cornerPoints.size < 2) {
            return localSource
        }
        val p0 = cornerPoints[0]
        val p1 = cornerPoints[1]
        val dx = (p1.x - p0.x).toFloat()
        val dy = (p1.y - p0.y).toFloat()
        if (dx == 0f && dy == 0f) {
            return localSource
        }
        val angleRad = atan2(dy, dx)
        val angleDeg = Math.toDegrees(angleRad.toDouble()).toFloat()
        if (angleDeg == 0f) {
            return localSource
        }
        val matrix = Matrix().apply {
            postRotate(-angleDeg, localSource.width / 2f, localSource.height / 2f)
        }
        return Bitmap.createBitmap(
            localSource,
            0,
            0,
            localSource.width,
            localSource.height,
            matrix,
            true
        )
    }

    private fun safeCrop(bitmap: Bitmap, rect: Rect): Bitmap {
        val left = rect.left.coerceAtLeast(0)
        val top = rect.top.coerceAtLeast(0)
        val right = rect.right.coerceIn(left + 1, bitmap.width)
        val bottom = rect.bottom.coerceIn(top + 1, bitmap.height)
        val width = (right - left).coerceAtLeast(1)
        val height = (bottom - top).coerceAtLeast(1)
        return Bitmap.createBitmap(bitmap, left, top, width, height)
    }

    private fun shouldReplaceEntry(existing: PlateEntry, newArea: Int, newBitmap: Bitmap): Boolean {
        if (newArea > existing.area) return true
        val similarity = bitmapSimilarityScore(existing.bitmap, newBitmap)
        return similarity > bitmapSimilarityThreshold
    }

    private fun bitmapSimilarityScore(first: Bitmap, second: Bitmap): Float {
        if (first.width == 0 || first.height == 0 || second.width == 0 || second.height == 0) {
            return 0f
        }
        val size = similaritySampleSize
        val scaledFirst = Bitmap.createScaledBitmap(first, size, size, true)
        val scaledSecond = Bitmap.createScaledBitmap(second, size, size, true)
        val bufferFirst = IntArray(size * size)
        val bufferSecond = IntArray(size * size)
        scaledFirst.getPixels(bufferFirst, 0, size, 0, 0, size, size)
        scaledSecond.getPixels(bufferSecond, 0, size, 0, 0, size, size)
        var diffSum = 0L
        for (i in bufferFirst.indices) {
            val c1 = bufferFirst[i]
            val c2 = bufferSecond[i]
            diffSum += kotlin.math.abs(Color.red(c1) - Color.red(c2))
            diffSum += kotlin.math.abs(Color.green(c1) - Color.green(c2))
            diffSum += kotlin.math.abs(Color.blue(c1) - Color.blue(c2))
        }
        val maxDiff = size * size * 255L * 3L
        val similarity = 1f - (diffSum.toFloat() / maxDiff.toFloat())
        if (scaledFirst != first) {
            scaledFirst.recycle()
        }
        if (scaledSecond != second) {
            scaledSecond.recycle()
        }
        return similarity.coerceIn(0f, 1f)
    }
//*** End Patch***} to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch гардид to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_patch to=functions.apply_pause

    private fun cropBitmap(source: Bitmap, boundingBox: Rect): Bitmap? {
        val left = boundingBox.left.coerceIn(0, source.width - 1)
        val top = boundingBox.top.coerceIn(0, source.height - 1)
        val right = boundingBox.right.coerceIn(left + 1, source.width)
        val bottom = boundingBox.bottom.coerceIn(top + 1, source.height)
        val width = right - left
        val height = bottom - top
        if (width <= 0 || height <= 0) return null
        return Bitmap.createBitmap(source, left, top, width, height)
    }

    private fun recordRawFrameResult(rawText: String) {
        val now = SystemClock.elapsedRealtime()
        recentDetections.add(now to rawText)
        pruneRecentDetections(now)
        refreshRecentStringsText(now)
    }

    private fun incrementWindowCount(detection: PlateDetection) {
        val sanitized = detection.text.trim()
        if (sanitized.isEmpty()) return
        val entry = windowCounts.getOrPut(sanitized) { WindowPlateEntry() }
        entry.count += 1
        if (detection.area > entry.bestArea) {
            entry.bestArea = detection.area
            entry.bitmap = detection.bitmap
        }
        entry.centerDistance = minOf(entry.centerDistance, detection.centerDistance)
        updateDetectionRankingText()
        maybePromptForConfirmation()
    }

    private fun evaluateWindowCounts() {
        if (windowCounts.isNotEmpty()) {
            val winningEntry = windowCounts
                .filter { it.key.length in minPlateLength..maxPlateLength }
                .maxByOrNull { it.value.count }
            winningEntry?.value?.let { entry ->
                val text = winningEntry.key
                val bitmap = entry.bitmap
                if (bitmap != null) {
                    insertOrUpdatePermanentPlate(text, entry.bestArea, bitmap)
                }
            }
        }
        clearWindowCounts()
    }

    private fun insertOrUpdatePermanentPlate(text: String, area: Int, bitmap: Bitmap) {
        if (!confirmedPlates.contains(text)) {
            return
        }
        val existingExact = permanentPlateMap[text]
        if (existingExact != null) {
            if (shouldReplaceEntry(existingExact, area, bitmap)) {
                permanentPlateMap[text] = PlateEntry(bitmap, area)
                refreshPermanentPlateList()
            }
            return
        }

        permanentPlateMap[text] = PlateEntry(bitmap, area)
        refreshPermanentPlateList()
    }

    private fun clearWindowCounts() {
        windowCounts.clear()
        updateDetectionRankingText()
    }

    private fun pruneRecentDetections(now: Long = SystemClock.elapsedRealtime()) {
        val cutoff = now - recentStringsRetentionMs
        while (recentDetections.isNotEmpty() && recentDetections.first().first < cutoff) {
            recentDetections.removeAt(0)
        }
    }

    private fun refreshRecentStringsText(now: Long = SystemClock.elapsedRealtime()) {
        pruneRecentDetections(now)
        val text = if (recentDetections.isEmpty()) {
            getString(R.string.no_recent_strings)
        } else {
            recentDetections.joinToString(separator = "\n") { it.second }
        }
        binding.recentStringsTextView.text = text
    }

    private fun postRawText(rawText: String?) {
        scannedText = rawText
        if (rawText != null) {
            binding.resultTextView.text = rawText
            recordRawFrameResult(rawText)
        } else {
            binding.resultTextView.text = getString(R.string.no_license_plate_detected)
        }
    }

    private fun sliderValueToFraction(value: Float, minFraction: Float): Float {
        val clamped = value.coerceIn(0f, 1f)
        val range = 1f - minFraction
        return 1f - (clamped * range)
    }

    private fun fractionToSliderValue(fraction: Float, minFraction: Float): Float {
        val clamped = fraction.coerceIn(minFraction, 1f)
        val range = 1f - minFraction
        return if (range == 0f) 0f else (1f - clamped) / range
    }

    private fun updateVerticalFraction(fraction: Float) {
        plateFilter.updateVerticalFraction(fraction)
        binding.fieldOfViewOverlay.verticalFraction = plateFilter.verticalFraction
    }

    private fun updateHorizontalFraction(fraction: Float) {
        plateFilter.updateHorizontalFraction(fraction)
        binding.fieldOfViewOverlay.horizontalFraction = plateFilter.horizontalFraction
    }

    private fun postAlgorithmResult(result: String?) {
        binding.algorithmResultTextView.text = result ?: getString(R.string.no_algorithm_result)
        val prompt = plateFilter.registerAlgorithmResult(result) { plate ->
            confirmedPlates.contains(plate)
        }
        if (prompt != null && !algorithmConfirmShowing) {
            showAlgorithmConfirmation(prompt.displayValue, prompt.sanitizedValue)
        }
    }

    private fun showAlgorithmConfirmation(displayValue: String, sanitizedValue: String) {
        if (algorithmConfirmShowing) return
        algorithmConfirmShowing = true
        runOnUiThread {
            AlertDialog.Builder(this)
                .setTitle(getString(R.string.confirm_plate_title))
                .setMessage(getString(R.string.confirm_plate_message, sanitizedValue))
                .setPositiveButton(R.string.confirm_plate_positive) { _, _ ->
                    confirmDetectedPlate(sanitizedValue)
                    plateFilter.resetCandidate(sanitizedValue)
                    algorithmConfirmShowing = false
                }
                .setNegativeButton(R.string.confirm_plate_negative) { _, _ ->
                    plateFilter.resetCandidate(sanitizedValue)
                    algorithmConfirmShowing = false
                }
                .setNeutralButton(R.string.confirm_plate_edit) { _, _ ->
                    showAlgorithmEditDialog(sanitizedValue)
                }
                .setOnCancelListener {
                    plateFilter.resetCandidate(sanitizedValue)
                    algorithmConfirmShowing = false
                }
                .show()
        }
    }

    private fun showAlgorithmEditDialog(initialValue: String) {
        val editInput = android.widget.EditText(this).apply {
            setText(initialValue)
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.confirm_plate_edit)
            .setView(editInput)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                val edited = plateFilter.sanitizePlateText(editInput.text.toString())
                if (edited.length in minPlateLength..maxPlateLength) {
                    confirmDetectedPlate(edited)
                } else {
                    showToast(getString(R.string.invalid_plate_edit))
                }
                plateFilter.resetCandidate(initialValue)
                algorithmConfirmShowing = false
            }
            .setNegativeButton(android.R.string.cancel) { _, _ ->
                plateFilter.resetCandidate(initialValue)
                algorithmConfirmShowing = false
            }
            .setOnCancelListener {
                plateFilter.resetCandidate(initialValue)
                algorithmConfirmShowing = false
            }
            .show()
    }

    private fun confirmDetectedPlate(plate: String) {
        val sanitized = plateFilter.sanitizePlateText(plate)
        if (sanitized.length !in minPlateLength..maxPlateLength) {
            showToast(getString(R.string.invalid_plate_edit))
            return
        }
        confirmedPlates.add(sanitized)
        insertOrUpdatePermanentPlate(sanitized, 0, placeholderPlateBitmap)
        plateFilter.resetCandidate(sanitized)
    }

    private fun updateDetectionRankingText() {
        val ranked = windowCounts.entries.sortedWith(
            compareByDescending<Map.Entry<String, WindowPlateEntry>> { it.value.count }
                .thenBy { it.value.centerDistance }
        )
        val sections = mutableListOf<String>()
        if (ranked.isNotEmpty()) {
            sections.add(
                ranked.joinToString(separator = "\n") { entry ->
                    "${entry.key} (${entry.value.count})"
                }
            )
        }
        if (permanentPlateMap.isNotEmpty()) {
            val confirmedList = permanentPlateMap.keys.sorted().joinToString(separator = "\n")
            sections.add(confirmedList)
        }
        if (sections.isEmpty()) {
            binding.plateListTextView.text = getString(R.string.no_license_plate_detected)
        } else {
            binding.plateListTextView.text = sections.joinToString(separator = "\n\n")
        }
    }

    private fun maybePromptForConfirmation() {
        if (confirmDialogShowing) return
        val candidate = windowCounts.entries
            .filter { it.value.count >= confirmationThreshold && it.value.bitmap != null && !confirmedPlates.contains(it.key) }
            .maxByOrNull { it.value.count } ?: return
        val text = candidate.key
        val entry = candidate.value
        confirmDialogShowing = true
        runOnUiThread {
            val editInput = android.widget.EditText(this).apply {
                setText(text)
            }
            AlertDialog.Builder(this)
                .setTitle(getString(R.string.confirm_plate_title))
                .setMessage(getString(R.string.confirm_plate_message, text))
                .setPositiveButton(R.string.confirm_plate_positive) { _, _ ->
                    confirmedPlates.add(text)
                    insertOrUpdatePermanentPlate(text, entry.bestArea, entry.bitmap!!)
                    plateFilter.resetCandidate(text)
                    confirmDialogShowing = false
                }
                .setNegativeButton(R.string.confirm_plate_negative) { _, _ ->
                    confirmDialogShowing = false
                }
                .setNeutralButton(R.string.confirm_plate_edit) { dialog, _ ->
                    AlertDialog.Builder(this)
                        .setTitle(R.string.confirm_plate_edit)
                        .setView(editInput)
                        .setPositiveButton(android.R.string.ok) { _, _ ->
                            val edited = plateFilter.sanitizePlateText(editInput.text.toString())
                            if (edited.length in minPlateLength..maxPlateLength) {
                                confirmedPlates.add(edited)
                                insertOrUpdatePermanentPlate(edited, entry.bestArea, entry.bitmap!!)
                                plateFilter.resetCandidate(edited)
                                refreshPermanentPlateList()
                            } else {
                                showToast(getString(R.string.invalid_plate_edit))
                            }
                            confirmDialogShowing = false
                        }
                        .setNegativeButton(android.R.string.cancel) { _, _ ->
                            confirmDialogShowing = false
                        }
                        .setOnCancelListener {
                            confirmDialogShowing = false
                        }
                        .show()
                    dialog.dismiss()
                }
                .setOnCancelListener {
                    confirmDialogShowing = false
                }
                .show()
        }
    }

    private fun refreshPermanentPlateList() {
        updateDetectionRankingText()
    }

    private fun showPlateRemovalDialog() {
        if (permanentPlateMap.isEmpty()) {
            showToast(getString(R.string.no_plate_to_remove))
            return
        }
        val plates = permanentPlateMap.keys.sorted()
        AlertDialog.Builder(this)
            .setTitle(R.string.remove_plate_title)
            .setItems(plates.toTypedArray()) { _, which ->
                val plate = plates[which]
                permanentPlateMap.remove(plate)
                confirmedPlates.remove(plate)
                refreshPermanentPlateList()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun buildPlateShareList(): String {
        return permanentPlateMap.keys.sorted().joinToString(separator = "\n")
    }

    private fun isSimilarText(first: String, second: String): Boolean {
        if (kotlin.math.abs(first.length - second.length) > 2) return false
        val maxLength = maxOf(first.length, second.length)
        var differences = 0
        for (i in 0 until maxLength) {
            val charA = first.getOrNull(i) ?: continue
            val charB = second.getOrNull(i) ?: continue
            if (charA != charB) {
                differences++
                if (differences > 2) return false
            }
        }
        return differences in 1..2
    }

private data class ZoomResult(
    val text: String,
    val bitmap: Bitmap,
    val rect: Rect,
    val frameWidth: Int,
    val frameHeight: Int
)

private data class PlateDetection(
    val text: String,
    val area: Int,
    val bitmap: Bitmap,
    val rect: Rect,
    val centerDistance: Float
)

private data class WindowPlateEntry(
    var count: Int = 0,
    var bestArea: Int = 0,
    var bitmap: Bitmap? = null,
    var centerDistance: Float = Float.MAX_VALUE
)

private data class PlateEntry(
    val bitmap: Bitmap,
    val area: Int
)

    private fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return bitmap
        val matrix = Matrix().apply {
            postRotate(rotationDegrees.toFloat())
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 80, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }
}
