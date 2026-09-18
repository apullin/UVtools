/*
 *                     GNU AFFERO GENERAL PUBLIC LICENSE
 *                       Version 3, 19 November 2007
 *  Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 *  Everyone is permitted to copy and distribute verbatim copies
 *  of this license document, but changing it is not allowed.
 */

using System;
using System.Buffers;
using System.Drawing;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using CommunityToolkit.HighPerformance;
using Emgu.CV;
using EmguExtensions;

namespace UVtools.Core.Layers;

/// <summary>
/// One horizontal run of foreground pixels: row <see cref="Y"/>, columns <see cref="Start"/> up to but excluding <see cref="End"/>.
/// </summary>
public readonly record struct Run(int Y, int Start, int End)
{
    public int Length => End - Start;
}

/// <summary>
/// A binary image stored as horizontal foreground runs, with direct row lookup and run-based connected
/// component labeling.
/// </summary>
public sealed class RunLengthImage
{
    public int Width { get; }
    public int Height { get; }

    /// <summary>
    /// Number of runs in the image.
    /// </summary>
    public int RunCount { get; private set; }

    // Runs of row y are the indexes [_rowStart[y], _rowStart[y + 1])
    private readonly int[] _rowStart;
    private int[] _row;
    private int[] _start;
    private int[] _end;

    private RunLengthImage(int width, int height, int runCapacity)
    {
        Width = width;
        Height = height;
        _rowStart = new int[height + 1];
        _row = new int[runCapacity];
        _start = new int[runCapacity];
        _end = new int[runCapacity];
    }

    /// <summary>
    /// Gets the run at <paramref name="runIndex"/>.
    /// </summary>
    public Run this[int runIndex] => new(_row[runIndex], _start[runIndex], _end[runIndex]);


    /// <summary>
    /// Gets the runs of row <paramref name="y"/> as parallel spans of start (inclusive) and end (exclusive) columns.
    /// </summary>
    public void RowRuns(int y, out ReadOnlySpan<int> starts, out ReadOnlySpan<int> ends)
    {
        var first = _rowStart[y];
        var count = _rowStart[y + 1] - first;
        starts = _start.AsSpan(first, count);
        ends = _end.AsSpan(first, count);
    }

    /// <summary>
    /// Starts an image to be filled with <see cref="AddRun"/> in raster order, then sealed with <see cref="Complete"/>.
    /// </summary>
    internal static RunLengthImage CreateEmpty(int width, int height, int runCapacity) => new(width, height, Math.Max(1, runCapacity));

    private int _nextRow;

    /// <summary>
    /// Appends a run. Rows must not decrease between calls and runs within a row must be sorted and disjoint.
    /// </summary>
    internal void AddRun(int y, int start, int end)
    {
        while (_nextRow <= y) _rowStart[_nextRow++] = RunCount;
        Append(y, start, end);
    }

    /// <summary>
    /// Seals an image built with <see cref="AddRun"/>.
    /// </summary>
    internal void Complete()
    {
        while (_nextRow <= Height) _rowStart[_nextRow++] = RunCount;
    }


    /// <summary>
    /// Builds the runs of a grayscale image. A pixel is foreground when it is strictly above
    /// <paramref name="threshold"/>, the same rule as <c>THRESH_BINARY</c>, so a threshold of 0 keeps every lit pixel.
    /// </summary>
    public static RunLengthImage FromThreshold(ReadOnlySpan2D<byte> pixels, byte threshold)
    {
        var image = new RunLengthImage(pixels.Width, pixels.Height, Math.Max(64, pixels.Height * 4));
        for (var y = 0; y < pixels.Height; y++)
        {
            image._rowStart[y] = image.RunCount;
            image.ScanRow(y, pixels.GetRowSpan(y), threshold, 0);
        }

        image._rowStart[pixels.Height] = image.RunCount;
        return image;
    }

    /// <summary>
    /// Builds the runs of a crop placed at <paramref name="offset"/> inside an image of <paramref name="size"/>;
    /// everything outside the crop is background. The crop must lie inside the image.
    /// </summary>
    public static RunLengthImage FromThreshold(ReadOnlySpan2D<byte> pixels, byte threshold, Point offset, Size size)
    {
        if (offset.X < 0 || offset.Y < 0 || offset.X + pixels.Width > size.Width || offset.Y + pixels.Height > size.Height)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), $"A {pixels.Width}x{pixels.Height} crop at {offset} does not fit in {size}.");
        }

        var image = new RunLengthImage(size.Width, size.Height, Math.Max(64, pixels.Height * 4));
        for (var y = 0; y < size.Height; y++)
        {
            image._rowStart[y] = image.RunCount;
            var sourceY = y - offset.Y;
            if ((uint)sourceY < (uint)pixels.Height) image.ScanRow(y, pixels.GetRowSpan(sourceY), threshold, offset.X);
        }

        image._rowStart[size.Height] = image.RunCount;
        return image;
    }

    /// <summary>
    /// Appends the runs of one row. Whole vectors of background, or of foreground, are skipped in one
    /// compare each; only the pixels around a transition are looked at one by one.
    /// </summary>
    private void ScanRow(int y, ReadOnlySpan<byte> row, byte threshold, int columnOffset)
    {
        var width = row.Length;
        ref var first = ref MemoryMarshal.GetReference(row);
        var thresholdVector = new Vector<byte>(threshold);
        var lanes = Vector<byte>.Count;
        var x = 0;

        while (x < width)
        {
            // Background: skip vectors with nothing above the threshold
            while (x + lanes <= width && !Vector.GreaterThanAny(Vector.LoadUnsafe(ref first, (nuint)x), thresholdVector))
            {
                x += lanes;
            }

            while (x < width && Unsafe.Add(ref first, x) <= threshold) x++;
            if (x >= width) break;

            var start = x++;

            // Foreground: skip vectors that are entirely above the threshold
            while (x + lanes <= width && Vector.GreaterThanAll(Vector.LoadUnsafe(ref first, (nuint)x), thresholdVector))
            {
                x += lanes;
            }

            while (x < width && Unsafe.Add(ref first, x) > threshold) x++;

            Append(y, start + columnOffset, x + columnOffset);
        }
    }

    private void Append(int y, int start, int end)
    {
        if (RunCount == _start.Length)
        {
            Array.Resize(ref _row, RunCount * 2);
            Array.Resize(ref _start, RunCount * 2);
            Array.Resize(ref _end, RunCount * 2);
        }

        _row[RunCount] = y;
        _start[RunCount] = start;
        _end[RunCount] = end;
        RunCount++;
    }

    /// <summary>
    /// Bounding rectangle of the foreground, empty when there is none.
    /// </summary>
    public Rectangle Bounds
    {
        get
        {
            if (RunCount == 0) return Rectangle.Empty;
            int minX = int.MaxValue, maxX = 0, minY = _row[0], maxY = _row[RunCount - 1];
            for (var i = 0; i < RunCount; i++)
            {
                if (_start[i] < minX) minX = _start[i];
                if (_end[i] > maxX) maxX = _end[i];
            }

            return new Rectangle(minX, minY, maxX - minX, maxY - minY + 1);
        }
    }

    /// <summary>
    /// Paints the foreground into an 8-bit single-channel mat, at <paramref name="offset"/>, as 255.
    /// </summary>
    public void DrawInto(Mat mat, Point offset)
    {
        var span = mat.GetSpan2DOfBytes();
        for (var i = 0; i < RunCount; i++)
        {
            var y = _row[i] + offset.Y;
            if ((uint)y >= (uint)span.Height) continue;
            var start = Math.Max(0, _start[i] + offset.X);
            var end = Math.Min(span.Width, _end[i] + offset.X);
            if (end > start) span.GetRowSpan(y).Slice(start, end - start).Fill(byte.MaxValue);
        }
    }

    /// <summary>
    /// Returns pixels where <paramref name="current"/> minus <paramref name="previous"/> exceeds 127.
    /// Inputs contain only their respective bounding rectangles; the result uses <paramref name="window"/>
    /// coordinates.
    /// </summary>
    public static RunLengthImage Overhang(ReadOnlySpan2D<byte> current, Rectangle currentBounds,
        ReadOnlySpan2D<byte> previous, Rectangle previousBounds, Rectangle window)
    {
        if (current.Width != currentBounds.Width || current.Height != currentBounds.Height)
            throw new ArgumentException("Current pixels do not match their bounds.", nameof(current));
        if (!previousBounds.IsEmpty &&
            (previous.Width != previousBounds.Width || previous.Height != previousBounds.Height))
            throw new ArgumentException("Previous pixels do not match their bounds.", nameof(previous));
        if (!window.Contains(currentBounds) || (!previousBounds.IsEmpty && !window.Contains(previousBounds)))
            throw new ArgumentOutOfRangeException(nameof(window), $"{window} does not contain {currentBounds} and {previousBounds}.");

        var result = CreateEmpty(window.Width, window.Height, Math.Max(64, current.Height * 4));
        for (var y = 0; y < current.Height; y++)
        {
            var frameY = currentBounds.Y + y;
            var outputY = frameY - window.Y;
            var currentRow = current.GetRowSpan(y);
            var hasPreviousRow = !previousBounds.IsEmpty && frameY >= previousBounds.Top && frameY < previousBounds.Bottom;
            var previousRow = hasPreviousRow ? previous.GetRowSpan(frameY - previousBounds.Y) : default;
            var x = 0;
            var outputStart = -1;

            while (x < currentRow.Length)
            {
                var value = currentRow[x];
                var length = currentRow[x..].IndexOfAnyExcept(value);
                if (length < 0) length = currentRow.Length - x;
                var segmentStart = currentBounds.X + x;
                var segmentEnd = segmentStart + length;

                if (value <= 127)
                {
                    if (outputStart >= 0)
                    {
                        result.AddRun(outputY, outputStart - window.X, segmentStart - window.X);
                        outputStart = -1;
                    }
                }
                else
                {
                    while (segmentStart < segmentEnd)
                    {
                        byte previousValue;
                        int until;
                        if (!hasPreviousRow)
                        {
                            previousValue = 0;
                            until = segmentEnd;
                        }
                        else if (segmentStart < previousBounds.Left)
                        {
                            previousValue = 0;
                            until = Math.Min(segmentEnd, previousBounds.Left);
                        }
                        else if (segmentStart >= previousBounds.Right)
                        {
                            previousValue = 0;
                            until = segmentEnd;
                        }
                        else
                        {
                            var previousX = segmentStart - previousBounds.X;
                            previousValue = previousRow[previousX];
                            var previousLength = previousRow[previousX..].IndexOfAnyExcept(previousValue);
                            until = previousLength < 0
                                ? Math.Min(segmentEnd, previousBounds.Right)
                                : Math.Min(segmentEnd, segmentStart + previousLength);
                        }

                        if (value - previousValue > 127)
                        {
                            if (outputStart < 0) outputStart = segmentStart;
                        }
                        else if (outputStart >= 0)
                        {
                            result.AddRun(outputY, outputStart - window.X, segmentStart - window.X);
                            outputStart = -1;
                        }

                        segmentStart = until;
                    }
                }

                x += length;
            }

            if (outputStart >= 0)
                result.AddRun(outputY, outputStart - window.X, currentBounds.Right - window.X);
        }

        result.Complete();
        return result;
    }


    /// <summary>
    /// Erodes with a cross-shaped structuring element of the given radius, one iteration, and reflected
    /// borders (OpenCV's <c>MORPH_CROSS</c> kernel of size 2r+1 with <c>BORDER_REFLECT_101</c>). A pixel
    /// survives when the r pixels on each side of it in its row and in its column are all foreground, so
    /// the result is the horizontal shrink of every run intersected with the rows above and below.
    /// With reflected borders a run that touches an image edge keeps that end.
    /// </summary>
    public RunLengthImage ErodeCross(int radius)
    {
        if (radius <= 0) return this;
        var width = Width;
        var height = Height;
        var result = CreateEmpty(width, height, RunCount);
        if (RunCount == 0)
        {
            result.Complete();
            return result;
        }

        var maximumRowRuns = 0;
        for (var y = 0; y < height; y++) maximumRowRuns = Math.Max(maximumRowRuns, _rowStart[y + 1] - _rowStart[y]);
        var maximumIntervals = Math.Min(RunCount, checked(maximumRowRuns * (Math.Min(height, radius * 2 + 1))));
        var workspaceLength = checked(maximumIntervals * 4);
        int[]? rented = null;
        Span<int> workspace = workspaceLength <= 512
            ? stackalloc int[workspaceLength]
            : (rented = ArrayPool<int>.Shared.Rent(workspaceLength)).AsSpan(0, workspaceLength);
        var starts = workspace[..maximumIntervals];
        var ends = workspace.Slice(maximumIntervals, maximumIntervals);
        var scratchStarts = workspace.Slice(maximumIntervals * 2, maximumIntervals);
        var scratchEnds = workspace.Slice(maximumIntervals * 3, maximumIntervals);

        try
        {
            for (var y = 0; y < height; y++)
            {
                RowRuns(y, out var rowStarts, out var rowEnds);
                if (rowStarts.Length == 0) continue;

                // Horizontal shrink; an end on the image edge is mirrored and stays
                var count = 0;
                for (var i = 0; i < rowStarts.Length; i++)
                {
                    var s = rowStarts[i] == 0 ? 0 : rowStarts[i] + radius;
                    var e = rowEnds[i] == width ? width : rowEnds[i] - radius;
                    if (e <= s) continue;
                    starts[count] = s;
                    ends[count] = e;
                    count++;
                }

                // Vertical: intersect with every row within the radius, rows beyond an edge mirror back inside
                for (var d = -radius; d <= radius && count > 0; d++)
                {
                    if (d == 0) continue;
                    var other = Reflect101(y + d, height);
                    if (other == y) continue;
                    RowRuns(other, out var otherStarts, out var otherEnds);
                    count = Intersect(starts, ends, count, otherStarts, otherEnds, scratchStarts, scratchEnds);
                    var oldStarts = starts;
                    starts = scratchStarts;
                    scratchStarts = oldStarts;
                    var oldEnds = ends;
                    ends = scratchEnds;
                    scratchEnds = oldEnds;
                }

                for (var i = 0; i < count; i++) result.AddRun(y, starts[i], ends[i]);
            }
        }
        finally
        {
            if (rented is not null) ArrayPool<int>.Shared.Return(rented);
        }

        result.Complete();
        return result;
    }

    /// <summary>
    /// Maps an index outside [0, length) back inside the way OpenCV's BORDER_REFLECT_101 does, folding as
    /// many times as needed (gfedcb|abcdefgh|gfedcba), so a kernel taller than the image still behaves alike.
    /// </summary>
    private static int Reflect101(int index, int length)
    {
        if (length == 1) return 0;
        while (index < 0 || index >= length)
        {
            index = index < 0 ? -index : 2 * (length - 1) - index;
        }

        return index;
    }

    /// <summary>
    /// Intersects two sorted interval lists into the scratch arrays and returns the interval count.
    /// </summary>
    private static int Intersect(ReadOnlySpan<int> aStarts, ReadOnlySpan<int> aEnds, int aCount,
        ReadOnlySpan<int> bStarts, ReadOnlySpan<int> bEnds, Span<int> outStarts, Span<int> outEnds)
    {
        var n = 0;
        int i = 0, j = 0;
        while (i < aCount && j < bStarts.Length)
        {
            var lo = Math.Max(aStarts[i], bStarts[j]);
            var hi = Math.Min(aEnds[i], bEnds[j]);
            if (hi > lo)
            {
                outStarts[n] = lo;
                outEnds[n] = hi;
                n++;
            }

            if (aEnds[i] < bEnds[j]) i++;
            else j++;
        }

        return n;
    }

    /// <summary>
    /// Labels the connected components of the foreground with union-find over runs. Two runs on
    /// adjacent rows are connected when their column ranges overlap; with
    /// <paramref name="eightConnected"/> touching corners count as well.
    /// </summary>
    public RunLengthComponents LabelComponents(bool eightConnected)
    {
        var runCount = RunCount;
        var parent = new int[runCount];
        for (var i = 0; i < runCount; i++) parent[i] = i;

        var reach = eightConnected ? 1 : 0;
        for (var y = 1; y < Height; y++)
        {
            int i = _rowStart[y], iEnd = _rowStart[y + 1];
            int j = _rowStart[y - 1], jEnd = _rowStart[y];
            while (i < iEnd && j < jEnd)
            {
                if (_start[i] - reach < _end[j] && _start[j] < _end[i] + reach) Union(parent, i, j);
                if (_end[i] < _end[j]) i++;
                else j++;
            }
        }

        var runComponent = ArrayPool<int>.Shared.Rent(Math.Max(1, runCount));
        try
        {
            Array.Fill(runComponent, -1, 0, runCount);
            var count = 0;
            for (var i = 0; i < runCount; i++)
            {
                var root = Find(parent, i);
                parent[i] = root;
                if (runComponent[root] < 0) runComponent[root] = count++;
            }

            for (var i = runCount - 1; i >= 0; i--) runComponent[i] = runComponent[parent[i]];

            var area = new int[count];
            var runsPerComponent = new int[count + 1];
            for (var i = 0; i < runCount; i++) runsPerComponent[runComponent[i] + 1]++;
            for (var component = 0; component < count; component++)
                runsPerComponent[component + 1] += runsPerComponent[component];

            var componentRuns = parent;
            for (var i = 0; i < runCount; i++)
            {
                var component = runComponent[i];
                componentRuns[runsPerComponent[component] + area[component]++] = i;
            }

            Array.Clear(area);
            var bounds = new Rectangle[count];
            for (var i = 0; i < runCount; i++)
            {
                var component = runComponent[i];
                var y = _row[i];
                area[component] += _end[i] - _start[i];
                if (bounds[component].IsEmpty)
                {
                    bounds[component] = new Rectangle(_start[i], y, _end[i] - _start[i], 1);
                    continue;
                }

                bounds[component] = Rectangle.FromLTRB(
                    Math.Min(bounds[component].Left, _start[i]),
                    Math.Min(bounds[component].Top, y),
                    Math.Max(bounds[component].Right, _end[i]),
                    Math.Max(bounds[component].Bottom, y + 1));
            }

            return new RunLengthComponents(this, count, area, bounds, runsPerComponent, componentRuns);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(runComponent);
        }
    }

    private static int Find(int[] parent, int i)
    {
        while (parent[i] != i)
        {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }

        return i;
    }

    private static void Union(int[] parent, int a, int b)
    {
        a = Find(parent, a);
        b = Find(parent, b);
        if (a == b) return;
        // Keep the earlier run as the root so that roots appear in raster order
        if (a < b) parent[b] = a;
        else parent[a] = b;
    }
}

/// <summary>
/// The connected components of a <see cref="RunLengthImage"/>, numbered in raster order of their first run.
/// </summary>
public sealed class RunLengthComponents
{
    private readonly RunLengthImage _image;
    private readonly int[] _area;
    private readonly Rectangle[] _bounds;
    private readonly int[] _componentRunStart;
    private readonly int[] _componentRuns;

    internal RunLengthComponents(RunLengthImage image, int count, int[] area, Rectangle[] bounds,
        int[] componentRunStart, int[] componentRuns)
    {
        _image = image;
        Count = count;
        _area = area;
        _bounds = bounds;
        _componentRunStart = componentRunStart;
        _componentRuns = componentRuns;
    }

    /// <summary>
    /// Number of components. Background is not a component.
    /// </summary>
    public int Count { get; }

    /// <summary>
    /// Number of foreground pixels of a component.
    /// </summary>
    public int Area(int component) => _area[component];

    /// <summary>
    /// Bounding rectangle of a component.
    /// </summary>
    public Rectangle Bounds(int component) => _bounds[component];

    /// <summary>
    /// Run indexes of a component, in raster order.
    /// </summary>
    public ReadOnlySpan<int> RunsOf(int component) =>
        _componentRuns.AsSpan(_componentRunStart[component], _componentRunStart[component + 1] - _componentRunStart[component]);

    /// <summary>
    /// The run at <paramref name="runIndex"/> of the underlying image.
    /// </summary>
    public Run GetRun(int runIndex) => _image[runIndex];
}
