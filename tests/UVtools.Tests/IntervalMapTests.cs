using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using EmguExtensions;
using UVtools.Core.Layers;
using Xunit;

namespace UVtools.Tests;

public class IntervalMapTests
{
    private const int Width = 61;
    private const int Height = 23;

    /// <summary>
    /// A random binary image with blobby runs, as a pixel array and as runs.
    /// </summary>
    private static (bool[,] pixels, RunLengthImage runs) RandomImage(Random random, int width = Width, int height = Height, double density = 0.4)
    {
        var pixels = new bool[height, width];
        using var mat = new Mat(height, width, DepthType.Cv8U, 1);
        var span = mat.GetSpan<byte>();
        for (var y = 0; y < height; y++)
        {
            var x = 0;
            while (x < width)
            {
                var length = random.Next(1, 9);
                var lit = random.NextDouble() < density;
                for (var k = 0; k < length && x < width; k++, x++)
                {
                    pixels[y, x] = lit;
                    span[y * width + x] = lit ? (byte)255 : (byte)0;
                }
            }
        }

        return (pixels, RunLengthImage.FromThreshold(mat.GetReadOnlySpan2DOfBytes(), 0));
    }

    private static bool[,] ToPixels(IntervalMap map)
    {
        var pixels = new bool[map.Height, map.Width];
        for (var y = 0; y < map.Height; y++)
        {
            var row = map.Row(y);
            for (var i = 0; i < row.Length; i += 2)
            {
                Assert.True(row[i] < row[i + 1], "empty interval");
                if (i > 0) Assert.True(row[i] > row[i - 1], "intervals must be sorted, disjoint and not touching");
                for (var x = row[i]; x < row[i + 1]; x++) pixels[y, x] = true;
            }
        }

        return pixels;
    }

    private static void AssertSame(bool[,] expected, IntervalMap map)
    {
        var actual = ToPixels(map);
        for (var y = 0; y < Height; y++)
            for (var x = 0; x < Width; x++)
                Assert.True(expected[y, x] == actual[y, x], $"pixel ({x},{y}) expected {expected[y, x]}");
    }

    private static bool At(bool[,] pixels, int x, int y) =>
        y >= 0 && y < pixels.GetLength(0) && x >= 0 && x < pixels.GetLength(1) && pixels[y, x];

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void UnionSubtractAndIntersectionMatchPixelArithmetic(int seed)
    {
        var random = new Random(seed);
        var (basePixels, baseRuns) = RandomImage(random);
        var map = IntervalMap.FromImage(baseRuns);
        AssertSame(basePixels, map);

        var expected = (bool[,])basePixels.Clone();
        for (var round = 0; round < 40; round++)
        {
            // Operands are smaller than the map and placed at an offset, sometimes hanging over the edge
            var (opPixels, opRuns) = RandomImage(random, random.Next(1, 30), random.Next(1, 12), random.NextDouble());
            var opMap = IntervalMap.FromImage(opRuns);
            var offset = new Point(random.Next(-5, Width), random.Next(-3, Height));

            long expectedOverlap = 0;
            for (var y = 0; y < opRuns.Height; y++)
                for (var x = 0; x < opRuns.Width; x++)
                    if (opPixels[y, x] && At(expected, x + offset.X, y + offset.Y)) expectedOverlap++;
            Assert.Equal(expectedOverlap, map.CountIntersection(opMap, offset));

            var subtract = random.Next(2) == 0;
            for (var y = 0; y < opRuns.Height; y++)
                for (var x = 0; x < opRuns.Width; x++)
                {
                    if (!opPixels[y, x]) continue;
                    int tx = x + offset.X, ty = y + offset.Y;
                    if (ty < 0 || ty >= Height || tx < 0 || tx >= Width) continue;
                    expected[ty, tx] = !subtract;
                }

            if (subtract) map.Subtract(opMap, offset);
            else map.Union(opMap, offset);
            AssertSame(expected, map);
        }

    }

    [Fact]
    public void ComplementAndFullCoverEveryPixel()
    {
        var (pixels, runs) = RandomImage(new Random(7));
        var source = IntervalMap.FromImage(runs);
        var complement = source.Complement();
        var expected = new bool[Height, Width];
        for (var y = 0; y < Height; y++)
            for (var x = 0; x < Width; x++)
                expected[y, x] = !pixels[y, x];
        AssertSame(expected, complement);

        complement.Union(source);
        AssertSame(ToPixels(IntervalMap.Full(Width, Height)), complement);

        complement.Subtract(source);
        AssertSame(expected, complement);
        complement.SetAll();
        AssertSame(ToPixels(IntervalMap.Full(Width, Height)), complement);
    }

    [Theory]
    [InlineData(11)]
    [InlineData(12)]
    public void DirectThresholdComplementAndMapOperationsMatchPixels(int seed)
    {
        var random = new Random(seed);
        var (basePixels, baseRuns) = RandomImage(random);
        var map = IntervalMap.FromImage(baseRuns);
        AssertSame(basePixels, map.Clone());

        var complementPixels = new bool[Height, Width];
        for (var y = 0; y < Height; y++)
            for (var x = 0; x < Width; x++)
                complementPixels[y, x] = !basePixels[y, x];
        AssertSame(complementPixels, map.Complement());

        var (opPixels, opRuns) = RandomImage(random, 31, 13, 0.55);
        var opMap = IntervalMap.FromImage(opRuns);
        var offset = new Point(17, 5);
        var unionExpected = (bool[,])basePixels.Clone();
        var subtractExpected = (bool[,])basePixels.Clone();
        for (var y = 0; y < opRuns.Height; y++)
            for (var x = 0; x < opRuns.Width; x++)
            {
                if (!opPixels[y, x]) continue;
                unionExpected[y + offset.Y, x + offset.X] = true;
                subtractExpected[y + offset.Y, x + offset.X] = false;
            }

        var union = map.Clone();
        union.Union(opMap, offset);
        AssertSame(unionExpected, union);
        var subtract = map.Clone();
        subtract.Subtract(opMap, offset);
        AssertSame(subtractExpected, subtract);
    }

    [Theory]
    [InlineData(21)]
    [InlineData(22)]
    [InlineData(23)]
    public void FusedUnionThenSubtractMatchesSeparateOperations(int seed)
    {
        var random = new Random(seed);
        for (var round = 0; round < 40; round++)
        {
            var (_, currentRuns) = RandomImage(random, density: random.NextDouble());
            var (_, subtractRuns) = RandomImage(random, density: random.NextDouble());
            var (_, unionRuns) = RandomImage(random, density: random.NextDouble());
            var current = IntervalMap.FromImage(currentRuns);
            var subtract = IntervalMap.FromImage(subtractRuns);
            var union = IntervalMap.FromImage(unionRuns);

            var expected = current.Clone();
            expected.Union(union);
            expected.Subtract(subtract);

            current.UnionThenSubtract(union, subtract);
            AssertSame(ToPixels(expected), current);
        }
    }

    [Fact]
    public void TouchingIntervalsMergeAndClippingHoldsAtTheEdges()
    {
        var map = new IntervalMap(10, 1);
        var (_, runs) = RandomImage(new Random(1), 3, 1, 1.0); // [0,3)
        var operand = IntervalMap.FromImage(runs);

        map.Union(operand);
        map.Union(operand, new Point(3, 0)); // touching: becomes [0,6)
        Assert.Equal(new[] { 0, 6 }, map.Row(0).ToArray());

        map.Union(operand, new Point(8, 0)); // clipped to [8,10)
        map.Union(operand, new Point(-2, 0)); // clipped to [0,1), already covered
        map.Subtract(operand, new Point(2, 0)); // cuts [2,5)
        Assert.Equal(new[] { 0, 2, 5, 6, 8, 10 }, map.Row(0).ToArray());
        Assert.Equal(4, map.CountIntersection(operand, new Point(-1, 0)) +
                        map.CountIntersection(operand, new Point(7, 0)));

        map.Union(operand, new Point(0, 5)); // off the map: ignored
        Assert.Equal(new[] { 0, 2, 5, 6, 8, 10 }, map.Row(0).ToArray());
    }
}
