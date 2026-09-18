using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using EmguExtensions;
using UVtools.Core.Layers;
using Xunit;

namespace UVtools.Tests;

public class RunLengthImageTests
{
    private static Mat Blobs(int seed, int width, int height, int blobs, int maxSize)
    {
        var random = new Random(seed);
        var mat = new Mat(height, width, DepthType.Cv8U, 1);
        mat.SetTo(new MCvScalar(0));
        for (var i = 0; i < blobs; i++)
        {
            var w = random.Next(1, maxSize);
            var h = random.Next(1, maxSize);
            var x = random.Next(0, width - w);
            var y = random.Next(0, height - h);
            var brightness = random.Next(1, 256);
            if (random.Next(2) == 0)
                CvInvoke.Rectangle(mat, new Rectangle(x, y, w, h), new MCvScalar(brightness), -1);
            else
                CvInvoke.Ellipse(mat, new RotatedRect(new PointF(x + w / 2f, y + h / 2f), new SizeF(w, h), random.Next(0, 180)), new MCvScalar(brightness), -1);
        }

        return mat;
    }

    /// <summary>
    /// Components as OpenCV reports them: (area, bounds), as a sorted multiset.
    /// </summary>
    private static List<(int area, Rectangle bounds)> OpenCvComponents(Mat gray, byte threshold, bool eightConnected)
    {
        using var binary = new Mat();
        CvInvoke.Threshold(gray, binary, threshold, 255, ThresholdType.Binary);
        using var labels = new Mat();
        using var stats = new Mat();
        using var centroids = new Mat();
        var count = CvInvoke.ConnectedComponentsWithStats(binary, labels, stats, centroids,
            eightConnected ? LineType.EightConnected : LineType.FourConnected);
        var data = stats.GetSpan<int>();
        var result = new List<(int, Rectangle)>();
        for (var i = 1; i < count; i++)
        {
            var pos = i * stats.Cols;
            result.Add((data[pos + (int)ConnectedComponentsTypes.Area], new Rectangle(
                data[pos + (int)ConnectedComponentsTypes.Left], data[pos + (int)ConnectedComponentsTypes.Top],
                data[pos + (int)ConnectedComponentsTypes.Width], data[pos + (int)ConnectedComponentsTypes.Height])));
        }

        return Sorted(result);
    }

    private static List<(int area, Rectangle bounds)> RunComponents(Mat gray, byte threshold, bool eightConnected)
    {
        var runs = RunLengthImage.FromThreshold(gray.GetReadOnlySpan2DOfBytes(), threshold);
        var components = runs.LabelComponents(eightConnected);
        var result = new List<(int, Rectangle)>();
        for (var c = 0; c < components.Count; c++)
        {
            // The run list must account for exactly the component's area
            var fromRuns = 0;
            foreach (var runIndex in components.RunsOf(c)) fromRuns += components.GetRun(runIndex).Length;
            Assert.Equal(components.Area(c), fromRuns);
            result.Add((components.Area(c), components.Bounds(c)));
        }

        return Sorted(result);
    }

    private static List<(int area, Rectangle bounds)> Sorted(IEnumerable<(int area, Rectangle bounds)> items) =>
        items.OrderBy(i => i.bounds.Y).ThenBy(i => i.bounds.X).ThenBy(i => i.area).ThenBy(i => i.bounds.Width).ThenBy(i => i.bounds.Height).ToList();

    [Theory]
    [InlineData(1, false)]
    [InlineData(1, true)]
    [InlineData(2, false)]
    [InlineData(2, true)]
    [InlineData(3, false)]
    [InlineData(3, true)]
    public void LabelsTheSameComponentsAsOpenCv(int seed, bool eightConnected)
    {
        // Odd sizes so rows are not a multiple of the vector width, and a threshold that splits the brightness range
        using var mat = Blobs(seed, 613, 397, blobs: 400, maxSize: 24);
        foreach (var threshold in new byte[] { 0, 1, 100 })
        {
            var expected = OpenCvComponents(mat, threshold, eightConnected);
            var actual = RunComponents(mat, threshold, eightConnected);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void ThresholdIsStrict()
    {
        using var mat = new Mat(1, 8, DepthType.Cv8U, 1);
        var span = mat.GetSpan<byte>();
        new byte[] { 0, 1, 2, 1, 1, 255, 0, 1 }.CopyTo(span);

        var above0 = RunLengthImage.FromThreshold(mat.GetReadOnlySpan2DOfBytes(), 0);
        Assert.Equal(new[] { new Run(0, 1, 6), new Run(0, 7, 8) }, Enumerable.Range(0, above0.RunCount).Select(i => above0[i]));

        var above1 = RunLengthImage.FromThreshold(mat.GetReadOnlySpan2DOfBytes(), 1);
        Assert.Equal(new[] { new Run(0, 2, 3), new Run(0, 5, 6) }, Enumerable.Range(0, above1.RunCount).Select(i => above1[i]));

        Assert.Equal(0, RunLengthImage.FromThreshold(mat.GetReadOnlySpan2DOfBytes(), 255).RunCount);
    }

    [Fact]
    public void RunsCoverFullRowsAndSurviveVectorBoundaries()
    {
        // A solid image is a single run per row; a checkerboard of 1-pixel columns is width/2 runs per row
        using var solid = new Mat(3, 1000, DepthType.Cv8U, 1);
        solid.SetTo(new MCvScalar(255));
        var solidRuns = RunLengthImage.FromThreshold(solid.GetReadOnlySpan2DOfBytes(), 0);
        Assert.Equal(3, solidRuns.RunCount);
        Assert.Equal(new Run(2, 0, 1000), solidRuns[2]);
        Assert.Equal(1, solidRuns.LabelComponents(false).Count);

        using var stripes = new Mat(2, 1000, DepthType.Cv8U, 1);
        var span = stripes.GetSpan<byte>();
        for (var i = 0; i < span.Length; i++) span[i] = (byte)(i % 2 == 0 ? 200 : 0);
        var stripeRuns = RunLengthImage.FromThreshold(stripes.GetReadOnlySpan2DOfBytes(), 0);
        Assert.Equal(1000, stripeRuns.RunCount);
        // Stripes line up vertically, so each column is one component
        Assert.Equal(500, stripeRuns.LabelComponents(false).Count);
    }

    [Fact]
    public void ACropPlacedAtAnOffsetScansLikeThePaddedImage()
    {
        // A crop of the image placed back at its offset inside a larger frame must give the same runs as
        // scanning the larger frame directly
        using var frame = Blobs(21, 200, 120, blobs: 40, maxSize: 25);
        var crop = new Rectangle(37, 18, 121, 77);
        using var cropped = new Mat(frame, crop);
        using var padded = new Mat(frame.Size, DepthType.Cv8U, 1);
        padded.SetTo(new MCvScalar(0));
        using (var target = new Mat(padded, crop)) cropped.CopyTo(target);

        foreach (var threshold in new byte[] { 0, 90 })
        {
            var expected = RunLengthImage.FromThreshold(padded.GetReadOnlySpan2DOfBytes(), threshold);
            var actual = RunLengthImage.FromThreshold(cropped.GetReadOnlySpan2DOfBytes(), threshold, crop.Location, frame.Size);
            Assert.Equal(expected.RunCount, actual.RunCount);
            Assert.Equal(expected.Bounds, actual.Bounds);
            for (var i = 0; i < expected.RunCount; i++) Assert.Equal(expected[i], actual[i]);
        }

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            RunLengthImage.FromThreshold(cropped.GetReadOnlySpan2DOfBytes(), 0, new Point(100, 100), frame.Size));
    }

    [Fact]
    public void DiagonalTouchOnlyConnectsWhenEightConnected()
    {
        using var mat = new Mat(2, 2, DepthType.Cv8U, 1);
        var span = mat.GetSpan<byte>();
        new byte[] { 255, 0, 0, 255 }.CopyTo(span);
        var runs = RunLengthImage.FromThreshold(mat.GetReadOnlySpan2DOfBytes(), 0);
        Assert.Equal(2, runs.LabelComponents(false).Count);
        Assert.Equal(1, runs.LabelComponents(true).Count);
        Assert.Equal(new Rectangle(0, 0, 2, 2), runs.LabelComponents(true).Bounds(0));
    }
}
