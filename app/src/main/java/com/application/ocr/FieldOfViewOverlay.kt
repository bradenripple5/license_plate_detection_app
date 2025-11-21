package com.application.ocr

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class FieldOfViewOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    var verticalFraction: Float = 1f
        set(value) {
            val clamped = value.coerceIn(0f, 1f)
            if (field != clamped) {
                field = clamped
                invalidate()
            }
        }

    var horizontalFraction: Float = 1f
        set(value) {
            val clamped = value.coerceIn(0f, 1f)
            if (field != clamped) {
                field = clamped
                invalidate()
            }
        }

    private val shadePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#80000000")
    }

    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = resources.displayMetrics.density * 2
        color = Color.WHITE
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (verticalFraction >= 0.999f && horizontalFraction >= 0.999f) {
            return
        }
        val widthF = width.toFloat()
        val heightF = height.toFloat()
        val visibleHeight = heightF * verticalFraction
        val visibleWidth = widthF * horizontalFraction
        val top = (heightF - visibleHeight) / 2f
        val bottom = top + visibleHeight
        val left = (widthF - visibleWidth) / 2f
        val right = left + visibleWidth

        if (top > 0f) {
            canvas.drawRect(0f, 0f, widthF, top, shadePaint)
        }
        if (bottom < heightF) {
            canvas.drawRect(0f, bottom, widthF, heightF, shadePaint)
        }
        if (left > 0f) {
            canvas.drawRect(0f, top, left, bottom, shadePaint)
        }
        if (right < widthF) {
            canvas.drawRect(right, top, widthF, bottom, shadePaint)
        }
        canvas.drawRect(left, top, right, bottom, borderPaint)
    }
}
