/*
 *                     GNU AFFERO GENERAL PUBLIC LICENSE
 *                       Version 3, 19 November 2007
 *  Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 *  Everyone is permitted to copy and distribute verbatim copies
 *  of this license document, but changing it is not allowed.
 */

using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using EmguExtensions;

namespace UVtools.Core.Layers;

/// <summary>
/// A layer's decoded bounding-box image prepared for run-based issue detection.
/// </summary>
internal sealed class LayerRunImage : IDisposable
{
    private readonly Rectangle _bounds;
    private Mat? _pixels;
    private bool _disposed;
    private LayerRunImage(Mat? pixels, Rectangle bounds)
    {
        _pixels = pixels;
        _bounds = bounds;
    }

    public static LayerRunImage FromLayer(Layer layer)
    {
        var compressed = layer.CompressedMat;
        if (compressed.IsEmpty) return new LayerRunImage(null, Rectangle.Empty);
        var bounds = compressed.Roi.Size.IsEmpty
            ? new Rectangle(0, 0, compressed.Width, compressed.Height)
            : compressed.Roi;
        return new LayerRunImage(compressed.RawDecompress(), bounds);
    }

    public RunLengthImage Runs(byte threshold, Rectangle window)
    {
        ThrowIfDisposed();
        if (_pixels is null) return Empty(window.Size);
        var offset = new Point(_bounds.X - window.X, _bounds.Y - window.Y);
        return RunLengthImage.FromThreshold(_pixels.GetReadOnlySpan2DOfBytes(), threshold, offset, window.Size);
    }

    public IntervalMap Intervals(byte threshold, Rectangle window, bool complement = false)
    {
        ThrowIfDisposed();
        if (_pixels is null)
            return complement ? IntervalMap.Full(window.Width, window.Height) : new IntervalMap(window.Width, window.Height);
        var offset = new Point(_bounds.X - window.X, _bounds.Y - window.Y);
        return IntervalMap.FromThreshold(_pixels.GetReadOnlySpan2DOfBytes(), threshold, offset, window.Size, complement);
    }

    public void ThresholdInto(Mat destination, byte threshold, Rectangle window)
    {
        ThrowIfDisposed();
        if (_pixels is null) return;
        if (destination.Size != window.Size || !window.Contains(_bounds))
            throw new ArgumentException($"Destination {window} does not contain layer bounds {_bounds}.", nameof(window));

        var targetBounds = new Rectangle(_bounds.X - window.X, _bounds.Y - window.Y, _bounds.Width, _bounds.Height);
        using var target = new Mat(destination, targetBounds);
        CvInvoke.Threshold(_pixels, target, threshold, byte.MaxValue, ThresholdType.Binary);
    }

    public RunLengthImage Overhang(LayerRunImage previous, Rectangle window)
    {
        ArgumentNullException.ThrowIfNull(previous);
        ThrowIfDisposed();
        previous.ThrowIfDisposed();
        if (_pixels is null) return Empty(window.Size);
        var pixels = _pixels.GetReadOnlySpan2DOfBytes();
        return previous._pixels is null
            ? RunLengthImage.Overhang(pixels, _bounds, default, Rectangle.Empty, window)
            : RunLengthImage.Overhang(pixels, _bounds, previous._pixels.GetReadOnlySpan2DOfBytes(), previous._bounds, window);
    }

    public Mat DecompressRectangle(Rectangle rectangle)
    {
        ThrowIfDisposed();
        var result = EmguCvExtensions.InitMat(rectangle.Size);
        if (_pixels is null) return result;
        var intersection = Rectangle.Intersect(_bounds, rectangle);
        if (intersection.IsEmpty) return result;

        var source = _pixels.GetReadOnlySpan2DOfBytes();
        var destination = result.GetSpan2DOfBytes();
        var sourceX = intersection.X - _bounds.X;
        var destinationX = intersection.X - rectangle.X;
        for (var y = intersection.Top; y < intersection.Bottom; y++)
        {
            source.GetRowSpan(y - _bounds.Y).Slice(sourceX, intersection.Width)
                .CopyTo(destination.GetRowSpan(y - rectangle.Y).Slice(destinationX, intersection.Width));
        }

        return result;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    private static RunLengthImage Empty(Size size)
    {
        var image = RunLengthImage.CreateEmpty(size.Width, size.Height, 1);
        image.Complete();
        return image;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _pixels?.Dispose();
        _pixels = null;
    }
}
