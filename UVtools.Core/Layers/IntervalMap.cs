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

namespace UVtools.Core.Layers;

/// <summary>
/// A mutable binary image stored as sorted, disjoint column intervals per row. Set operations merge sorted
/// rows instead of scanning pixels. Resin-trap detection uses it to fold solid and air masks between layers.
/// </summary>
public sealed class IntervalMap
{
    public int Width { get; }
    public int Height { get; }

    // Immutable maps use two compact arrays. Rows are copied out lazily only when this map is mutated.
    // This matters for resin detection, which retains two maps per layer: a jagged array there creates
    // thousands of tiny arrays per layer and puts far more pressure on the GC than the interval data itself.
    private int[]? _packedIntervals;
    private int[]? _packedRowStarts;
    private int[]?[]? _rows;
    private int[]? _counts;
    private bool[]? _overridden;
    private int _overrideCount;
    private bool _baseFull;
    private int[]? _fullRow;
    private int[] _scratch = [];

    public IntervalMap(int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(width);
        ArgumentOutOfRangeException.ThrowIfNegative(height);
        Width = width;
        Height = height;
    }

    private IntervalMap(int width, int height, int[] packedIntervals, int[] packedRowStarts)
        : this(width, height)
    {
        _packedIntervals = packedIntervals;
        _packedRowStarts = packedRowStarts;
    }

    /// <summary>
    /// A map with the same foreground as <paramref name="image"/>.
    /// </summary>
    public static IntervalMap FromImage(RunLengthImage image)
    {
        var intervals = GC.AllocateUninitializedArray<int>(checked(image.RunCount * 2));
        var rowStarts = GC.AllocateUninitializedArray<int>(image.Height + 1);
        var n = 0;
        for (var y = 0; y < image.Height; y++)
        {
            rowStarts[y] = n;
            image.RowRuns(y, out var starts, out var ends);
            for (var i = 0; i < starts.Length; i++)
            {
                intervals[n++] = starts[i];
                intervals[n++] = ends[i];
            }
        }

        rowStarts[image.Height] = n;
        return new IntervalMap(image.Width, image.Height, intervals, rowStarts);
    }


    /// <summary>
    /// Builds an interval map directly from decoded grayscale pixels without first allocating a
    /// <see cref="RunLengthImage"/>. Pixels strictly above <paramref name="threshold"/> are foreground;
    /// when <paramref name="complement"/> is set, that result is inverted across the whole output image.
    /// </summary>
    public static IntervalMap FromThreshold(ReadOnlySpan2D<byte> pixels, byte threshold, Point offset, Size size,
        bool complement = false)
    {
        if (offset.X < 0 || offset.Y < 0 || offset.X + pixels.Width > size.Width || offset.Y + pixels.Height > size.Height)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), $"A {pixels.Width}x{pixels.Height} crop at {offset} does not fit in {size}.");
        }

        using var builder = CreateBuilder(size.Width, size.Height, Math.Max(16, pixels.Height * 4));
        for (var y = 0; y < size.Height; y++)
        {
            var sourceY = y - offset.Y;
            if ((uint)sourceY >= (uint)pixels.Height)
            {
                if (complement) builder.AppendInterval(y, 0, size.Width);
                continue;
            }

            ScanRow(builder, y, pixels.GetRowSpan(sourceY), threshold, offset.X, complement);
            if (complement && offset.X + pixels.Width < size.Width)
            {
                builder.AppendInterval(y, offset.X + pixels.Width, size.Width);
            }
        }

        return builder.Complete();
    }

    /// <summary>
    /// Returns the inverse of this map.
    /// </summary>
    public IntervalMap Complement()
    {
        using var builder = CreateBuilder(Width, Height, Math.Max(16, Height * 4));
        for (var y = 0; y < Height; y++)
        {
            var row = Row(y);
            var x = 0;
            for (var i = 0; i < row.Length; i += 2)
            {
                if (row[i] > x) builder.AppendInterval(y, x, row[i]);
                x = row[i + 1];
            }

            if (x < Width) builder.AppendInterval(y, x, Width);
        }

        return builder.Complete();
    }

    public IntervalMap Clone()
    {
        if (_overrideCount == 0)
        {
            if (_baseFull) return Full(Width, Height);
            return _packedRowStarts is null
                ? new IntervalMap(Width, Height)
                : new IntervalMap(Width, Height, _packedIntervals!, _packedRowStarts);
        }

        using var builder = CreateBuilder(Width, Height, Math.Max(16, Height * 4));
        for (var y = 0; y < Height; y++)
        {
            var row = Row(y);
            for (var i = 0; i < row.Length; i += 2) builder.AppendInterval(y, row[i], row[i + 1]);
        }

        return builder.Complete();
    }

    /// <summary>
    /// A map with every pixel foreground.
    /// </summary>
    public static IntervalMap Full(int width, int height)
    {
        var map = new IntervalMap(width, height);
        map.SetAll();
        return map;
    }

    /// <summary>
    /// Makes every pixel foreground.
    /// </summary>
    public void SetAll()
    {
        _packedIntervals = null;
        _packedRowStarts = null;
        _baseFull = Width > 0;
        if (_baseFull) _fullRow ??= [0, Width];
        if (_overridden is not null) Array.Clear(_overridden);
        _overrideCount = 0;
    }

    /// <summary>
    /// The intervals of row <paramref name="y"/> as interleaved start, end pairs.
    /// </summary>
    public ReadOnlySpan<int> Row(int y)
    {
        if (_overridden is not null && _overridden[y]) return _rows![y].AsSpan(0, _counts![y]);
        if (_baseFull) return _fullRow;
        if (_packedRowStarts is null) return default;
        var start = _packedRowStarts[y];
        return _packedIntervals.AsSpan(start, _packedRowStarts[y + 1] - start);
    }

    /// <summary>
    /// Whether pixel (<paramref name="x"/>, <paramref name="y"/>) is foreground.
    /// </summary>
    public bool Contains(int y, int x)
    {
        if ((uint)y >= (uint)Height) return false;
        var row = Row(y);
        int lo = 0, hi = row.Length / 2 - 1;
        while (lo <= hi)
        {
            var mid = (lo + hi) >> 1;
            if (row[2 * mid + 1] <= x) lo = mid + 1;
            else if (row[2 * mid] > x) hi = mid - 1;
            else return true;
        }

        return false;
    }

    /// <summary>
    /// Adds the foreground of another interval map, placed at <paramref name="offset"/>.
    /// </summary>
    public void Union(IntervalMap map, Point offset = default)
    {
        for (var y = 0; y < map.Height; y++)
        {
            var row = map.Row(y);
            if (row.Length == 0) continue;
            UnionRow(y + offset.Y, row, offset.X);
        }
    }

    /// <summary>
    /// Removes the foreground of another interval map, placed at <paramref name="offset"/>.
    /// </summary>
    public void Subtract(IntervalMap map, Point offset = default)
    {
        for (var y = 0; y < map.Height; y++)
        {
            var row = map.Row(y);
            if (row.Length == 0) continue;
            SubtractRow(y + offset.Y, row, offset.X);
        }
    }

    /// <summary>
    /// Replaces this map with <c>(this ∪ union) ∖ subtract</c>. All three interval lists are folded in one
    /// row sweep, avoiding the intermediate row copies produced by separate <see cref="Union(IntervalMap, Point)"/>
    /// and <see cref="Subtract(IntervalMap, Point)"/> calls.
    /// </summary>
    public void UnionThenSubtract(IntervalMap union, IntervalMap subtract)
    {
        ArgumentNullException.ThrowIfNull(union);
        ArgumentNullException.ThrowIfNull(subtract);
        if (union.Width != Width || union.Height != Height)
        {
            throw new ArgumentException("The map dimensions must match.", nameof(union));
        }

        if (subtract.Width != Width || subtract.Height != Height)
        {
            throw new ArgumentException("The map dimensions must match.", nameof(subtract));
        }

        for (var y = 0; y < Height; y++)
        {
            UnionThenSubtractRow(y, union.Row(y), subtract.Row(y));
        }
    }

    /// <summary>
    /// Counts pixels that are foreground here and in <paramref name="map"/> at <paramref name="offset"/>.
    /// </summary>
    public long CountIntersection(IntervalMap map, Point offset = default)
    {
        long total = 0;
        for (var y = 0; y < map.Height; y++)
        {
            var intervals = map.Row(y);
            if (!intervals.IsEmpty) total += IntersectionRow(y + offset.Y, intervals, offset.X);
        }

        return total;
    }

    private void UnionRow(int y, ReadOnlySpan<int> intervals, int offsetX)
    {
        if ((uint)y >= (uint)Height) return;
        var row = Row(y);
        var scratch = EnsureScratch(row.Length + intervals.Length);
        var n = 0;
        int i = 0, j = 0;
        int currentStart = 0, currentEnd = int.MinValue;
        var open = false;

        while (i < row.Length || j < intervals.Length)
        {
            int s, e;
            if (j >= intervals.Length || (i < row.Length && row[i] <= intervals[j] + offsetX))
            {
                s = row[i];
                e = row[i + 1];
                i += 2;
            }
            else
            {
                s = Math.Max(0, intervals[j] + offsetX);
                e = Math.Min(Width, intervals[j + 1] + offsetX);
                j += 2;
                if (e <= s) continue;
            }

            if (open && s <= currentEnd)
            {
                if (e > currentEnd) currentEnd = e;
                continue;
            }

            if (open)
            {
                scratch[n++] = currentStart;
                scratch[n++] = currentEnd;
            }

            currentStart = s;
            currentEnd = e;
            open = true;
        }

        if (open)
        {
            scratch[n++] = currentStart;
            scratch[n++] = currentEnd;
        }

        StoreRow(y, scratch, n);
    }

    /// <summary>
    /// Removes sorted intervals from row <paramref name="y"/>.
    /// </summary>

    private void SubtractRow(int y, ReadOnlySpan<int> intervals, int offsetX)
    {
        if ((uint)y >= (uint)Height) return;
        var row = Row(y);
        if (row.Length == 0) return;
        var scratch = EnsureScratch(row.Length + intervals.Length);
        var n = 0;
        var j = 0;

        for (var i = 0; i < row.Length; i += 2)
        {
            var s = row[i];
            var e = row[i + 1];
            while (j < intervals.Length && intervals[j + 1] + offsetX <= s) j += 2;
            var k = j;
            while (k < intervals.Length && intervals[k] + offsetX < e)
            {
                var cutStart = intervals[k] + offsetX;
                var cutEnd = intervals[k + 1] + offsetX;
                if (cutStart > s)
                {
                    scratch[n++] = s;
                    scratch[n++] = cutStart;
                }

                s = Math.Max(s, cutEnd);
                if (s >= e) break;
                k += 2;
            }

            if (s < e)
            {
                scratch[n++] = s;
                scratch[n++] = e;
            }
        }

        StoreRow(y, scratch, n);
    }

    private void UnionThenSubtractRow(int y, ReadOnlySpan<int> union, ReadOnlySpan<int> subtract)
    {
        var row = Row(y);
        if (union.Length == 0)
        {
            if (subtract.Length > 0) SubtractRow(y, subtract, 0);
            return;
        }

        if (subtract.Length == 0)
        {
            UnionRow(y, union, 0);
            return;
        }

        var scratch = EnsureScratch(row.Length + subtract.Length + union.Length);
        var rowIndex = 0;
        var unionIndex = 0;
        var subtractIndex = 0;
        var n = 0;

        while (rowIndex < row.Length || unionIndex < union.Length)
        {
            int start, end;
            if (unionIndex >= union.Length || (rowIndex < row.Length && row[rowIndex] <= union[unionIndex]))
            {
                start = row[rowIndex];
                end = row[rowIndex + 1];
                rowIndex += 2;
            }
            else
            {
                start = union[unionIndex];
                end = union[unionIndex + 1];
                unionIndex += 2;
            }

            // Merge overlapping or touching intervals from this map and the union map.
            while (true)
            {
                var nextRowStart = rowIndex < row.Length ? row[rowIndex] : int.MaxValue;
                var nextUnionStart = unionIndex < union.Length ? union[unionIndex] : int.MaxValue;
                if (nextRowStart > end && nextUnionStart > end) break;

                if (nextRowStart <= nextUnionStart)
                {
                    if (row[rowIndex + 1] > end) end = row[rowIndex + 1];
                    rowIndex += 2;
                }
                else
                {
                    if (union[unionIndex + 1] > end) end = union[unionIndex + 1];
                    unionIndex += 2;
                }
            }

            while (subtractIndex < subtract.Length && subtract[subtractIndex + 1] <= start) subtractIndex += 2;
            var remainderStart = start;
            var cutIndex = subtractIndex;
            while (cutIndex < subtract.Length && subtract[cutIndex] < end)
            {
                var cutStart = subtract[cutIndex];
                var cutEnd = subtract[cutIndex + 1];
                if (cutStart > remainderStart)
                {
                    scratch[n++] = remainderStart;
                    scratch[n++] = Math.Min(cutStart, end);
                }

                if (cutEnd > remainderStart) remainderStart = cutEnd;
                if (remainderStart >= end) break;
                cutIndex += 2;
            }

            subtractIndex = cutIndex;
            if (remainderStart >= end) continue;
            scratch[n++] = remainderStart;
            scratch[n++] = end;
        }

        StoreRow(y, scratch, n);
    }

    /// <summary>
    /// Counts the overlap between row <paramref name="y"/> and one interval.
    /// </summary>
    public long IntersectionRow(int y, int start, int end)
    {
        if ((uint)y >= (uint)Height) return 0;
        var row = Row(y);
        long total = 0;
        for (var i = 0; i < row.Length; i += 2)
        {
            if (row[i] >= end) break;
            var lo = Math.Max(row[i], start);
            var hi = Math.Min(row[i + 1], end);
            if (hi > lo) total += hi - lo;
        }

        return total;
    }

    private long IntersectionRow(int y, ReadOnlySpan<int> intervals, int offsetX)
    {
        if ((uint)y >= (uint)Height) return 0;
        var row = Row(y);
        long total = 0;
        int i = 0, j = 0;
        while (i < row.Length && j < intervals.Length)
        {
            var shiftedEnd = intervals[j + 1] + offsetX;
            var lo = Math.Max(row[i], intervals[j] + offsetX);
            var hi = Math.Min(row[i + 1], shiftedEnd);
            if (hi > lo) total += hi - lo;
            if (row[i + 1] < shiftedEnd) i += 2;
            else j += 2;
        }

        return total;
    }


    private static void ScanRow(Builder builder, int y, ReadOnlySpan<byte> row, byte threshold, int offsetX,
        bool complement)
    {
        var width = row.Length;
        ref var first = ref MemoryMarshal.GetReference(row);
        var thresholdVector = new Vector<byte>(threshold);
        var lanes = Vector<byte>.Count;
        var x = 0;

        if (complement && offsetX > 0) builder.AppendInterval(y, 0, offsetX);

        while (x < width)
        {
            if (complement)
            {
                while (x + lanes <= width && Vector.GreaterThanAll(Vector.LoadUnsafe(ref first, (nuint)x), thresholdVector)) x += lanes;
                while (x < width && Unsafe.Add(ref first, x) > threshold) x++;
                if (x >= width) break;
                var start = x++;
                while (x + lanes <= width && !Vector.GreaterThanAny(Vector.LoadUnsafe(ref first, (nuint)x), thresholdVector)) x += lanes;
                while (x < width && Unsafe.Add(ref first, x) <= threshold) x++;
                builder.AppendInterval(y, start + offsetX, x + offsetX);
            }
            else
            {
                while (x + lanes <= width && !Vector.GreaterThanAny(Vector.LoadUnsafe(ref first, (nuint)x), thresholdVector)) x += lanes;
                while (x < width && Unsafe.Add(ref first, x) <= threshold) x++;
                if (x >= width) break;
                var start = x++;
                while (x + lanes <= width && Vector.GreaterThanAll(Vector.LoadUnsafe(ref first, (nuint)x), thresholdVector)) x += lanes;
                while (x < width && Unsafe.Add(ref first, x) > threshold) x++;
                builder.AppendInterval(y, start + offsetX, x + offsetX);
            }
        }
    }

    private int[] EnsureRow(int y, int capacity)
    {
        _rows ??= new int[]?[Height];
        _counts ??= new int[Height];
        _overridden ??= new bool[Height];
        var row = _rows[y];
        if (!_overridden[y])
        {
            var source = Row(y);
            row = new int[Math.Max(capacity, Math.Max(8, source.Length))];
            source.CopyTo(row);
            _rows[y] = row;
            _counts[y] = source.Length;
            _overridden[y] = true;
            _overrideCount++;
        }
        else if (row is null || row.Length < capacity)
        {
            row = new int[Math.Max(capacity, row is null ? 8 : row.Length * 2)];
            _rows[y] = row;
        }

        return row;
    }

    private int[] EnsureScratch(int capacity)
    {
        if (_scratch.Length < capacity) _scratch = new int[Math.Max(capacity, _scratch.Length * 2)];
        return _scratch;
    }

    private void StoreRow(int y, int[] scratch, int n)
    {
        if (n == 0)
        {
            StoreEmptyRow(y);
            return;
        }

        var row = EnsureRow(y, n);
        Array.Copy(scratch, row, n);
        _counts![y] = n;
    }


    private void StoreEmptyRow(int y)
    {
        _rows ??= new int[]?[Height];
        _counts ??= new int[Height];
        _overridden ??= new bool[Height];
        if (!_overridden[y])
        {
            _overridden[y] = true;
            _overrideCount++;
        }

        _rows[y] ??= [];
        _counts[y] = 0;
    }

    internal static Builder CreateBuilder(int width, int height, int initialIntervalCapacity = 0) =>
        new(width, height, initialIntervalCapacity);

    /// <summary>
    /// Builds the compact immutable base of an interval map in row order. Its temporary interval storage is
    /// pooled; only the exact-size packed interval and row-index arrays survive <see cref="Complete"/>.
    /// </summary>
    internal sealed class Builder : IDisposable
    {
        private readonly int _width;
        private readonly int _height;
        private int[] _buffer;
        private int[]? _rowStarts;
        private int _count;
        private int _nextRow;
        private bool _completed;

        internal Builder(int width, int height, int initialIntervalCapacity)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(width);
            ArgumentOutOfRangeException.ThrowIfNegative(height);
            ArgumentOutOfRangeException.ThrowIfNegative(initialIntervalCapacity);
            _width = width;
            _height = height;
            _rowStarts = GC.AllocateUninitializedArray<int>(height + 1);
            _buffer = ArrayPool<int>.Shared.Rent(Math.Max(16, checked(initialIntervalCapacity * 2)));
        }

        internal void AppendInterval(int y, int start, int end)
        {
            ObjectDisposedException.ThrowIf(_buffer.Length == 0, this);
            if ((uint)y >= (uint)_height) return;
            if (y < _nextRow - 1) throw new InvalidOperationException("Intervals must be appended in row order.");
            while (_nextRow <= y) _rowStarts![_nextRow++] = _count;

            start = Math.Max(0, start);
            end = Math.Min(_width, end);
            if (end <= start) return;

            var rowStart = _rowStarts![y];
            if (_count > rowStart && start <= _buffer[_count - 1])
            {
                if (end > _buffer[_count - 1]) _buffer[_count - 1] = end;
                return;
            }

            EnsureCapacity(_count + 2);
            _buffer[_count++] = start;
            _buffer[_count++] = end;
        }

        internal IntervalMap Complete()
        {
            ObjectDisposedException.ThrowIf(_buffer.Length == 0, this);
            if (_completed) throw new InvalidOperationException("The interval map builder is already complete.");
            while (_nextRow <= _height) _rowStarts![_nextRow++] = _count;
            var intervals = _count == 0 ? [] : GC.AllocateUninitializedArray<int>(_count);
            _buffer.AsSpan(0, _count).CopyTo(intervals);
            var rowStarts = _rowStarts!;
            _rowStarts = null;
            _completed = true;
            return new IntervalMap(_width, _height, intervals, rowStarts);
        }

        private void EnsureCapacity(int required)
        {
            if (required <= _buffer.Length) return;
            var next = ArrayPool<int>.Shared.Rent(Math.Max(required, checked(_buffer.Length * 2)));
            _buffer.AsSpan(0, _count).CopyTo(next);
            ArrayPool<int>.Shared.Return(_buffer);
            _buffer = next;
        }

        public void Dispose()
        {
            if (_buffer.Length == 0) return;
            ArrayPool<int>.Shared.Return(_buffer);
            _buffer = [];
            _rowStarts = null;
        }
    }
}
