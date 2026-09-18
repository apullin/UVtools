using System;
using System.Drawing;
using System.Linq;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using EmguExtensions;
using UVtools.Core.Layers;
using Xunit;

namespace UVtools.Tests;

/// <summary>
/// The overhang detector's erosion, mask and contour tracing on runs must reproduce OpenCV pixel for pixel.
/// </summary>
public class RunMorphologyTests
{
    private static Mat Blobs(int seed, int width, int height, int blobs, int maxSize, bool binary)
    {
        var random = new Random(seed);
        var mat = new Mat(height, width, DepthType.Cv8U, 1);
        mat.SetTo(new MCvScalar(0));
        for (var i = 0; i < blobs; i++)
        {
            var w = random.Next(1, maxSize);
            var h = random.Next(1, maxSize);
            // Allow blobs to hang over the edges so that runs touch the image border
            var x = random.Next(-maxSize / 2, width);
            var y = random.Next(-maxSize / 2, height);
            var value = binary ? 255 : random.Next(1, 256);
            CvInvoke.Rectangle(mat, new Rectangle(x, y, w, h), new MCvScalar(value), -1);
        }

        return mat;
    }

    private static RunLengthImage Runs(Mat mat, byte threshold) => RunLengthImage.FromThreshold(mat.GetReadOnlySpan2DOfBytes(), threshold);

    private static void AssertSamePixels(Mat expected, RunLengthImage actual)
    {
        using var painted = new Mat(expected.Size, DepthType.Cv8U, 1);
        painted.SetTo(new MCvScalar(0));
        actual.DrawInto(painted, Point.Empty);
        using var diff = new Mat();
        CvInvoke.AbsDiff(expected, painted, diff);
        Assert.Equal(0, CvInvoke.CountNonZero(diff));
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 3)]
    [InlineData(3, 7)]
    [InlineData(4, 12)]
    public void ErodeCrossMatchesOpenCv(int seed, int radius)
    {
        using var mat = Blobs(seed, 173, 131, blobs: 60, maxSize: 40, binary: true);
        using var kernel = CvInvoke.GetStructuringElement(MorphShapes.Cross, new Size(2 * radius + 1, 2 * radius + 1), new Point(-1, -1));
        using var expected = new Mat();
        CvInvoke.Erode(mat, expected, kernel, new Point(-1, -1), 1, BorderType.Default, default);

        AssertSamePixels(expected, Runs(mat, 0).ErodeCross(radius));
    }

    [Fact]
    public void ErodeCrossKeepsEdgesUnderReflection()
    {
        // A full-width band survives horizontally at both edges; a band touching only the top keeps its top rows
        using var mat = new Mat(30, 50, DepthType.Cv8U, 1);
        mat.SetTo(new MCvScalar(0));
        CvInvoke.Rectangle(mat, new Rectangle(0, 10, 50, 9), new MCvScalar(255), -1);
        CvInvoke.Rectangle(mat, new Rectangle(20, 0, 9, 8), new MCvScalar(255), -1);
        using var kernel = CvInvoke.GetStructuringElement(MorphShapes.Cross, new Size(7, 7), new Point(-1, -1));
        using var expected = new Mat();
        CvInvoke.Erode(mat, expected, kernel, new Point(-1, -1), 1, BorderType.Default, default);
        AssertSamePixels(expected, Runs(mat, 0).ErodeCross(3));
    }

    [Theory]
    [InlineData(2, 40)]   // two rows, kernel far taller than the image: reflection folds many times
    [InlineData(5, 12)]
    [InlineData(41, 40)]  // one row more than the radius
    [InlineData(1, 3)]    // a single row reflects onto itself
    public void ErodeCrossMatchesOpenCvWhenTheImageIsShorterThanTheKernel(int height, int radius)
    {
        using var mat = Blobs(height * 7 + radius, 300, height, blobs: 30, maxSize: 90, binary: true);
        using var kernel = CvInvoke.GetStructuringElement(MorphShapes.Cross, new Size(2 * radius + 1, 2 * radius + 1), new Point(-1, -1));
        using var expected = new Mat();
        CvInvoke.Erode(mat, expected, kernel, new Point(-1, -1), 1, BorderType.Default, default);
        AssertSamePixels(expected, Runs(mat, 0).ErodeCross(radius));
    }


    [Theory]
    [InlineData(5)]
    [InlineData(6)]
    public void OverhangMatchesSubtractAndThreshold(int seed)
    {
        var frame = new Size(400, 300);
        using var current = new Mat(frame, DepthType.Cv8U, 1);
        using var previous = new Mat(frame, DepthType.Cv8U, 1);
        current.SetTo(new MCvScalar(0));
        previous.SetTo(new MCvScalar(0));
        using (var blobs = Blobs(seed, 260, 180, blobs: 40, maxSize: 60, binary: false))
        using (var target = new Mat(current, new Rectangle(60, 40, blobs.Width, blobs.Height))) blobs.CopyTo(target);
        using (var blobs = Blobs(seed + 7, 300, 150, blobs: 40, maxSize: 60, binary: false))
        using (var target = new Mat(previous, new Rectangle(20, 90, blobs.Width, blobs.Height))) blobs.CopyTo(target);

        using var expected = new Mat();
        CvInvoke.Subtract(current, previous, expected);
        CvInvoke.Threshold(expected, expected, 127, 255, ThresholdType.Binary);
        var currentBounds = CvInvoke.BoundingRectangle(current);
        var previousBounds = CvInvoke.BoundingRectangle(previous);
        var window = Rectangle.Union(currentBounds, previousBounds);
        using var currentCrop = new Mat(current, currentBounds);
        using var previousCrop = new Mat(previous, previousBounds);
        using var expectedCrop = new Mat(expected, window);

        AssertSamePixels(expectedCrop, RunLengthImage.Overhang(currentCrop.GetReadOnlySpan2DOfBytes(), currentBounds,
            previousCrop.GetReadOnlySpan2DOfBytes(), previousBounds, window));
    }

    [Fact]
    public void ContoursOfTheCroppedRasterMatchTheFullImage()
    {
        using var mat = Blobs(9, 300, 200, blobs: 25, maxSize: 30, binary: true);
        using var kernel = CvInvoke.GetStructuringElement(MorphShapes.Cross, new Size(5, 5), new Point(-1, -1));
        using var eroded = new Mat();
        CvInvoke.Erode(mat, eroded, kernel, new Point(-1, -1), 1, BorderType.Default, default);
        var offset = new Point(1000, 2000);

        using var expected = eroded.FindContours(out var expectedHierarchy, RetrType.Tree, ChainApproxMethod.ChainApproxSimple, offset);

        var runs = Runs(mat, 0).ErodeCross(2);
        var bounds = runs.Bounds;
        Assert.Equal(CvInvoke.BoundingRectangle(eroded), bounds);
        using var raster = EmguCvExtensions.InitMat(new Size(bounds.Width + 2, bounds.Height + 2));
        runs.DrawInto(raster, new Point(1 - bounds.X, 1 - bounds.Y));
        using var actual = raster.FindContours(out var actualHierarchy, RetrType.Tree, ChainApproxMethod.ChainApproxSimple,
            new Point(offset.X + bounds.X - 1, offset.Y + bounds.Y - 1));

        Assert.Equal(expected.Size, actual.Size);
        Assert.Equal(expectedHierarchy, actualHierarchy);
        for (var i = 0; i < expected.Size; i++)
        {
            Assert.Equal(expected[i].ToArray(), actual[i].ToArray());
        }
    }

    [Fact]
    public void BoundsAndDrawRoundTrip()
    {
        using var mat = Blobs(11, 120, 80, blobs: 15, maxSize: 20, binary: true);
        var runs = Runs(mat, 0);
        Assert.Equal(CvInvoke.BoundingRectangle(mat), runs.Bounds);
        AssertSamePixels(mat, runs);
        Assert.Equal(Rectangle.Empty, RunLengthImage.FromThreshold(mat.GetReadOnlySpan2DOfBytes(), 255).Bounds);
    }
}
