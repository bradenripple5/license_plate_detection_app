package com.application.ocr

import android.graphics.Rect
import com.google.mlkit.vision.text.Text
import kotlin.math.max
import kotlin.math.sqrt

class PlateFilter(
    private val minPlateLength: Int,
    private val maxPlateLength: Int,
    val minVerticalFraction: Float = 0.3f,
    val minHorizontalFraction: Float = 0.3f,
    private val algorithmConfirmationThreshold: Int = 3
) {

    data class AlgorithmPrompt(val displayValue: String, val sanitizedValue: String)

    var verticalFraction: Float = 1f
        private set
    var horizontalFraction: Float = 1f
        private set
    var previewWidth: Int = 0
        private set
    var previewHeight: Int = 0
        private set

    private val algorithmDetectionCounts = mutableMapOf<String, Int>()
    private val algorithmPrompted = mutableSetOf<String>()
    private val commonWords = setOf(
        "ALABAMA",
        "ALASKA",
        "ARIZONA",
        "ARKANSAS",
        "CALIFORNIA",
        "COLORADO",
        "CONNECTICUT",
        "DELAWARE",
        "FLORIDA",
        "GEORGIA",
        "HAWAII",
        "IDAHO",
        "ILLINOIS",
        "INDIANA",
        "IOWA",
        "KANSAS",
        "KENTUCKY",
        "LOUISIANA",
        "MAINE",
        "MARYLAND",
        "MASSACHUSETTS",
        "MICHIGAN",
        "MINNESOTA",
        "MISSISSIPPI",
        "MISSOURI",
        "MONTANA",
        "NEBRASKA",
        "NEVADA",
        "NEW HAMPSHIRE",
        "NEW JERSEY",
        "NEW MEXICO",
        "NEW YORK",
        "NORTH CAROLINA",
        "NORTH DAKOTA",
        "OHIO",
        "OKLAHOMA",
        "OREGON",
        "PENNSYLVANIA",
        "RHODE ISLAND",
        "SOUTH CAROLINA",
        "SOUTH DAKOTA",
        "TENNESSEE",
        "TEXAS",
        "UTAH",
        "VERMONT",
        "VIRGINIA",
        "WASHINGTON",
        "WEST VIRGINIA",
        "WISCONSIN",
        "WYOMING"
    ).map { sanitizePlateText(it) }.toSet()

    fun sanitizePlateText(text: String): String {
        return text.uppercase()
            .filter { it.isLetterOrDigit() }
            .trim()
    }

    fun updatePreviewSize(width: Int, height: Int) {
        previewWidth = width
        previewHeight = height
    }

    fun updateVerticalFraction(fraction: Float) {
        verticalFraction = fraction.coerceIn(minVerticalFraction, 1f)
    }

    fun updateHorizontalFraction(fraction: Float) {
        horizontalFraction = fraction.coerceIn(minHorizontalFraction, 1f)
    }

    fun filterVisibleText(result: Text, imageWidth: Int, imageHeight: Int): String? {
        if (imageHeight <= 0 || imageWidth <= 0) {
            return result.text.takeIf { it.isNotBlank() }
        }
        val window = computeImageWindowBounds(imageWidth, imageHeight)
        val linesInWindow = mutableListOf<String>()
        result.textBlocks.forEach { block ->
            block.lines.forEach { line ->
                val box = line.boundingBox ?: return@forEach
                val centerY = box.centerY().toFloat() / imageHeight
                val centerX = box.centerX().toFloat() / imageWidth
                if (centerY in window.top..window.bottom && centerX in window.left..window.right) {
                    val content = line.text.trim()
                    if (content.isNotEmpty()) {
                        linesInWindow.add(content)
                    }
                }
            }
        }
        if (linesInWindow.isEmpty()) {
            return null
        }
        return linesInWindow.joinToString(separator = "\n")
    }

    fun computeAlgorithmResult(result: Text, imageWidth: Int, imageHeight: Int): String? {
        if (imageWidth <= 0 || imageHeight <= 0) {
            return result.text.takeIf { it.isNotBlank() }
        }
        val candidates = result.textBlocks.flatMap { block ->
            block.lines.mapNotNull { line ->
                val box = line.boundingBox ?: return@mapNotNull null
                val textValue = line.text.trim()
                if (textValue.isEmpty()) null else AlgorithmCandidate(textValue, box)
            }
        }
        if (candidates.isEmpty()) {
            return null
        }
        var top = 1f / 3f
        var bottom = 2f / 3f
        val minHeight = 0.05f
        val shrinkStep = 0.05f
        val prioritizedCandidates = candidates.filter { candidate ->
            val sanitized = sanitizePlateText(candidate.text)
            sanitized.length in 4..7 && sanitized.isNotEmpty() && !commonWords.contains(sanitized)
        }
        if (prioritizedCandidates.isNotEmpty()) {
            val best = prioritizedCandidates.minByOrNull {
                computeCenterDistance(it.rect, imageWidth, imageHeight)
            }
            if (best != null) {
                return best.text
            }
        }
        repeat(20) {
            val inWindow = candidates.filter { candidate ->
                val centerY = candidate.rect.centerY().toFloat() / imageHeight
                centerY in top..bottom
            }
            val unique = inWindow.map { it.text }.distinct()
            if (unique.size == 1) {
                return unique.first()
            }
            if (unique.isEmpty() && it == 0) {
                return candidates.first().text
            }
            val height = bottom - top
            if (height <= minHeight) {
                return unique.firstOrNull() ?: candidates.first().text
            }
            top += shrinkStep / 2f
            bottom -= shrinkStep / 2f
            if (top >= bottom) {
                return unique.firstOrNull() ?: candidates.first().text
            }
        }
        return candidates.first().text
    }

    fun registerAlgorithmResult(
        rawResult: String?,
        isAlreadyConfirmed: (String) -> Boolean
    ): AlgorithmPrompt? {
        if (rawResult.isNullOrBlank()) return null
        val sanitized = sanitizePlateText(rawResult)
        if (sanitized.isBlank()) {
            return null
        }
        if (isAlreadyConfirmed(sanitized)) {
            resetCandidate(sanitized)
            return null
        }
        if (sanitized.length !in minPlateLength..maxPlateLength) {
            return null
        }
        val newCount = (algorithmDetectionCounts[sanitized] ?: 0) + 1
        algorithmDetectionCounts[sanitized] = newCount
        if (newCount >= algorithmConfirmationThreshold && algorithmPrompted.add(sanitized)) {
            return AlgorithmPrompt(rawResult, sanitized)
        }
        return null
    }

    fun resetCandidate(sanitized: String) {
        algorithmDetectionCounts.remove(sanitized)
        algorithmPrompted.remove(sanitized)
    }

    fun computeCenterDistance(rect: Rect, frameWidth: Int, frameHeight: Int): Float {
        val cx = rect.centerX().toFloat() / frameWidth
        val cy = rect.centerY().toFloat() / frameHeight
        val dx = cx - 0.5f
        val dy = cy - 0.5f
        return sqrt(dx * dx + dy * dy)
    }

    private fun computeImageWindowBounds(imageWidth: Int, imageHeight: Int): NormalizedWindow {
        var top = windowTopRatio()
        var bottom = windowBottomRatio()
        var left = windowLeftRatio()
        var right = windowRightRatio()
        val viewW = previewWidth
        val viewH = previewHeight
        if (viewW <= 0 || viewH <= 0 || imageWidth <= 0 || imageHeight <= 0) {
            return NormalizedWindow(left, top, right, bottom)
        }
        val imageWidthF = imageWidth.toFloat()
        val imageHeightF = imageHeight.toFloat()
        val viewWF = viewW.toFloat()
        val viewHF = viewH.toFloat()
        val scale = max(viewWF / imageWidthF, viewHF / imageHeightF)
        val displayedWidth = imageWidthF * scale
        val displayedHeight = imageHeightF * scale
        val cropX = (displayedWidth - viewWF) / 2f
        val cropY = (displayedHeight - viewHF) / 2f

        fun convertHorizontal(fraction: Float): Float {
            val px = fraction * viewWF + cropX
            val imagePx = px / scale
            return (imagePx / imageWidthF).coerceIn(0f, 1f)
        }

        fun convertVertical(fraction: Float): Float {
            val px = fraction * viewHF + cropY
            val imagePx = px / scale
            return (imagePx / imageHeightF).coerceIn(0f, 1f)
        }

        left = convertHorizontal(left)
        right = convertHorizontal(right)
        top = convertVertical(top)
        bottom = convertVertical(bottom)
        return NormalizedWindow(left, top, right, bottom)
    }

    private fun windowTopRatio(): Float {
        return (1f - verticalFraction) / 2f
    }

    private fun windowBottomRatio(): Float {
        return 1f - windowTopRatio()
    }

    private fun windowLeftRatio(): Float {
        return (1f - horizontalFraction) / 2f
    }

    private fun windowRightRatio(): Float {
        return 1f - windowLeftRatio()
    }

    private data class AlgorithmCandidate(
        val text: String,
        val rect: Rect
    )

    private data class NormalizedWindow(
        val left: Float,
        val top: Float,
        val right: Float,
        val bottom: Float
    )
}
