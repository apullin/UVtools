using CommunityToolkit.HighPerformance;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.Drawing;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using EmguExtensions;
using UVtools.Core.Extensions;
using UVtools.Core.FileFormats;
using UVtools.Core.Layers;
using UVtools.Core.Operations;
using UVtools.Core.PixelEditor;
using ZLinq;

namespace UVtools.Core.Managers;

public sealed class IssueManager : RangeObservableCollection<MainIssue>
{
    private long _revision;

    public FileFormat SlicerFile { get; }

    /// <summary>
    /// Gets a monotonic revision that changes whenever the visible issue collection changes.
    /// </summary>
    public long Revision => Interlocked.Read(ref _revision);

    public List<MainIssue> IgnoredIssues { get; } = [];

    public bool HaveIssues => Count > 0;

    public IssueManager(FileFormat slicerFile)
    {
        SlicerFile = slicerFile;
    }

    protected override void OnCollectionChanged(NotifyCollectionChangedEventArgs e)
    {
        Interlocked.Increment(ref _revision);
        base.OnCollectionChanged(e);
    }

    /// <summary>
    /// Gets the visible <see cref="MainIssue"/> aka not ignored
    /// </summary>
    /// <returns></returns>
    public MainIssue[] GetVisible()
    {
        return this.AsValueEnumerable().Where(mainIssue => !IgnoredIssues.Contains(mainIssue)).ToArray();
    }

    public static Issue[] GetIssues(IEnumerable<MainIssue> issues)
    {
        var result = new List<Issue>();
        foreach (var mainIssue in issues)
        {
            result.AddRange(mainIssue);
        }

        return result.ToArray();
    }

    public static Issue[] GetIssuesBy(IEnumerable<MainIssue> issues, MainIssue.IssueType type, uint layerIndex)
    {
        var result = new List<Issue>();
        foreach (var mainIssue in issues)
        {
            if (mainIssue.Type != type) continue;
            if (!mainIssue.IsIssueInBetween(layerIndex)) continue;
            foreach (var issue in mainIssue)
            {
                if (issue.LayerIndex != layerIndex) continue;
                result.Add(issue);
            }
        }

        return result.ToArray();
    }

    public static Issue[] GetIssuesBy(IEnumerable<MainIssue> issues, MainIssue.IssueType type)
    {
        var result = new List<Issue>();
        foreach (var mainIssue in issues)
        {
            if (mainIssue.Type != type) continue;
            result.AddRange(mainIssue);
        }

        return result.ToArray();
    }


    public static Issue[] GetIssuesBy(IEnumerable<MainIssue> issues, uint layerIndex)
    {
        var result = new List<Issue>();
        foreach (var mainIssue in issues)
        {
            if (!mainIssue.IsIssueInBetween(layerIndex)) continue;
            foreach (var issue in mainIssue)
            {
                if (issue.LayerIndex != layerIndex) continue;
                result.Add(issue);
            }
        }

        return result.ToArray();
    }

    public Issue[] GetIssues()
    {
        return GetIssues(this);
    }

    public Issue[] GetIssuesBy(MainIssue.IssueType type)
    {
        return GetIssuesBy(this, type);
    }

    public Issue[] GetIssuesBy(MainIssue.IssueType type, uint layerIndex)
    {
        return GetIssuesBy(this, type, layerIndex);
    }

    public Issue[] GetIssuesBy(uint layerIndex)
    {
        return GetIssuesBy(this, layerIndex);
    }

    /// <summary>
    /// One hollow contour rasterized as intervals inside its bounding box, plus its overlap with known air.
    /// </summary>
    private readonly record struct HollowAirCheck(Rectangle Roi, IntervalMap? Fill, long OverlapCount);

    private sealed class DecodedLayerCache : IDisposable
    {
        private readonly Mat?[] _scratch = new Mat?[2];
        private uint _index;
        private LayerRunImage? _image;

        public LayerRunImage? Take(uint index)
        {
            if (_image is null || _index != index) return null;
            var image = _image;
            _image = null;
            return image;
        }

        public LayerRunImage GetOrDecode(Layer layer)
        {
            if (_image is not null && _index == layer.Index) return _image;
            _image?.Dispose();
            _image = LayerRunImage.FromLayer(layer);
            _index = layer.Index;
            return _image;
        }

        public void Keep(uint index, LayerRunImage image)
        {
            if (!ReferenceEquals(_image, image)) _image?.Dispose();
            _image = image;
            _index = index;
        }

        /// <summary>
        /// Returns a zeroed mat header over a worker-local buffer. Dispose the header, not the backing buffer.
        /// </summary>
        public Mat Scratch(Size size, int slot = 0)
        {
            var backing = _scratch[slot];
            var bytes = Math.Max(1L, (long)size.Width * size.Height);
            if (backing is null || (long)backing.Rows * backing.Cols < bytes)
            {
                backing?.Dispose();
                backing = new Mat(1, checked((int)bytes), DepthType.Cv8U, 1);
                _scratch[slot] = backing;
            }

            var mat = new Mat(size.Height, size.Width, DepthType.Cv8U, 1, backing.DataPointer, size.Width);
            mat.SetTo(new MCvScalar(0));
            return mat;
        }

        public void Dispose()
        {
            _image?.Dispose();
            _image = null;
            foreach (var scratch in _scratch) scratch?.Dispose();
            Array.Clear(_scratch);
        }
    }

    public List<MainIssue> DetectIssues(IssuesDetectionConfiguration? config = null, OperationProgress? progress = null)
    {
        if (SlicerFile.DecodeType == FileFormat.FileDecodeType.Partial) return [];

        config ??= new IssuesDetectionConfiguration();
        var (
            islandConfig,
            overhangConfig,
            resinTrapConfig,
            touchBoundConfig,
            printHeightConfig,
            emptyLayerConfig
            ) = config;

        progress ??= new OperationProgress();

        var result = new ConcurrentBag<MainIssue>();
        var resinTraps = new List<VectorOfVectorOfPoint>?[SlicerFile.LayerCount];
        var suctionCups = new List<VectorOfVectorOfPoint>?[SlicerFile.LayerCount];
        var externalContours = new VectorOfVectorOfPoint?[SlicerFile.LayerCount];
        var hollows = new List<VectorOfVectorOfPoint>?[SlicerFile.LayerCount];
        var airContours = new List<VectorOfVectorOfPoint>?[SlicerFile.LayerCount];
        // Per-layer interval maps for the resin trap passes, produced by the parallel pass so that the
        // sequential passes never touch a pixel: the solid at the drain threshold, and the air outside the part
        var solidMaps = new IntervalMap?[SlicerFile.LayerCount];
        var layerAirMaps = new IntervalMap?[SlicerFile.LayerCount];
        var resinTrapsContoursArea = new double[SlicerFile.LayerCount][];

        bool IsIgnored(MainIssue issue) => IgnoredIssues.Count > 0 && IgnoredIssues.Contains(issue);

        bool AddIssue(MainIssue issue)
        {
            if (IsIgnored(issue)) return false;
            result.Add(issue);
            return true;
        }

        List<MainIssue> GetResult()
        {
            return result.AsValueEnumerable().OrderBy(mainIssue => mainIssue.Type)
                .ThenBy(issue => issue.StartLayerIndex).ThenByDescending(issue => issue.Area).ToList();
        }

        // Gets the rectangle that encloses every contour of the group, clamped to bounds.
        // The resin trap passes only ever touch pixels inside this rectangle, so all the per-contour
        // mat work can be confined to it instead of running over the whole layer.
        static Rectangle GetContourGroupRoi(VectorOfVectorOfPoint group, Size bounds)
        {
            if (group.Size == 0) return Rectangle.Empty;

            var rect = CvInvoke.BoundingRectangle(group[0]);
            for (var i = 1; i < group.Size; i++)
            {
                rect = Rectangle.Union(rect, CvInvoke.BoundingRectangle(group[i]));
            }

            rect.Intersect(new Rectangle(Point.Empty, bounds));
            return rect;
        }

        // Rasterize with OpenCV to preserve contour-fill semantics, then keep the result as intervals.
        static HollowAirCheck RasterizeGroup(VectorOfVectorOfPoint group, Size bounds)
        {
            var roi = GetContourGroupRoi(group, bounds);
            if (roi.IsEmpty) return default;

            using var fillMat = EmguCvExtensions.InitMat(roi.Size);
            CvInvoke.DrawContours(fillMat, group, -1, EmguCvExtensions.WhiteColor, -1,
                LineType.EightConnected, null, int.MaxValue, new Point(-roi.X, -roi.Y));
            return new HollowAirCheck(roi,
                IntervalMap.FromThreshold(fillMat.GetReadOnlySpan2DOfBytes(), 0, Point.Empty, roi.Size), 0);
        }

        static HollowAirCheck CheckHollowAgainstAirMap(VectorOfVectorOfPoint group, IntervalMap airMap)
        {
            var check = RasterizeGroup(group, new Size(airMap.Width, airMap.Height));
            return check.Fill is null
                ? check
                : check with { OverlapCount = airMap.CountIntersection(check.Fill, check.Roi.Location) };
        }

        // True when any contour of the group on this layer or the one above overlaps contour.
        static bool GroupTouches(List<(VectorOfVectorOfPoint contour, uint layerIndex)> group, VectorOfVectorOfPoint contour, int layerIndex)
        {
            for (var i = group.Count - 1; i >= 0; i--)
            {
                if (group[i].layerIndex > layerIndex + 1) continue;
                if (EmguContours.ContoursIntersect(group[i].contour, contour)) return true;
            }
            return false;
        }

        static bool IssueGroupTouches(List<IssueOfContours> group, VectorOfVectorOfPoint contour, Rectangle contourBounds, int layerIndex)
        {
            for (var i = group.Count - 1; i >= 0; i--)
            {
                if (group[i].LayerIndex > layerIndex + 1) continue;
                // Most candidates are nowhere near: reject on the stored rectangles before copying the
                // contour into a native vector and rasterizing both
                if (!group[i].BoundingRectangle.IntersectsWith(contourBounds)) continue;
                using var vec = new VectorOfVectorOfPoint(group[i].Contours);
                if (EmguContours.ContoursIntersect(contour, vec)) return true;
            }
            return false;
        }

        if (printHeightConfig.Enabled && SlicerFile.MachineZ > 0)
        {
            float printHeightWithOffset = Layer.RoundHeight(SlicerFile.MachineZ + printHeightConfig.Offset);
            if (SlicerFile.PrintHeight > printHeightWithOffset)
            {
                var issues = (from layer in SlicerFile
                              where layer.PositionZ > printHeightWithOffset
                              select new Issue(layer)).ToList();

                if (issues.Count > 0) AddIssue(new MainIssue(MainIssue.IssueType.PrintHeight, issues));
            }
        }

        if (emptyLayerConfig.Enabled)
        {
            var classifyByPosition = emptyLayerConfig.IgnoreStartingEmptyLayers ||
                                     emptyLayerConfig.IgnoreLooseEmptyLayers ||
                                     emptyLayerConfig.IgnoreEndingEmptyLayers;
            var firstNonEmptyLayerIndex = 0;
            var lastNonEmptyLayerIndex = SlicerFile.Count - 1;
            if (classifyByPosition)
            {
                while (firstNonEmptyLayerIndex < SlicerFile.Count && SlicerFile[firstNonEmptyLayerIndex].IsEmpty)
                {
                    firstNonEmptyLayerIndex++;
                }

                while (lastNonEmptyLayerIndex >= firstNonEmptyLayerIndex && SlicerFile[lastNonEmptyLayerIndex].IsEmpty)
                {
                    lastNonEmptyLayerIndex--;
                }
            }

            for (var layerIndex = 0; layerIndex < SlicerFile.Count; layerIndex++)
            {
                var layer = SlicerFile[layerIndex];
                if (!layer.IsEmpty) continue;

                if (!classifyByPosition)
                {
                    AddIssue(new MainIssue(MainIssue.IssueType.EmptyLayer, new Issue(layer)));
                    continue;
                }

                if (layerIndex < firstNonEmptyLayerIndex)
                {
                    if (!emptyLayerConfig.IgnoreStartingEmptyLayers)
                        AddIssue(new MainIssue(MainIssue.IssueType.EmptyLayer, new Issue(layer)));
                }
                else if (layerIndex > lastNonEmptyLayerIndex)
                {
                    if (!emptyLayerConfig.IgnoreEndingEmptyLayers)
                        AddIssue(new MainIssue(MainIssue.IssueType.EmptyLayer, new Issue(layer)));
                }
                else if (!emptyLayerConfig.IgnoreLooseEmptyLayers)
                {
                    AddIssue(new MainIssue(MainIssue.IssueType.EmptyLayer, new Issue(layer)));
                }
            }
        }

        if (islandConfig.Enabled || overhangConfig.Enabled || resinTrapConfig.Enabled || touchBoundConfig.Enabled)
        {
            progress.Reset(OperationProgress.StatusIslands, SlicerFile.LayerCount);

            var firstLayer = SlicerFile.FirstLayer;

            int overhangsIterations = overhangConfig.ErodeIterations;
            using var overhangsKernel = EmguCvExtensions.CreateDynamicKernel(ref overhangsIterations, MorphShapes.Cross);
            // CreateDynamicKernel turns the iteration count into one pass of a cross this many pixels wide on each side
            var overhangsRadius = Math.Max((int)overhangConfig.ErodeIterations, 1);

            // Pixels of rect that are lit on this layer but not on the previous one, eroded like the overhang pass does
            Mat IslandOverhangImage(Mat layerRoi, Mat previousRoi, Rectangle rect)
            {
                using var islandRoi = layerRoi.Roi(rect);
                using var previousIslandRoi = previousRoi.Roi(rect);
                var overhang = new Mat();
                CvInvoke.Subtract(islandRoi, previousIslandRoi, overhang);
                CvInvoke.Threshold(overhang, overhang, 127, 255, ThresholdType.Binary);
                CvInvoke.Erode(overhang, overhang, overhangsKernel, EmguCvExtensions.AnchorCenter, overhangsIterations, BorderType.Default, default);
                return overhang;
            }

            // Detect contours. Each worker keeps its last decoded layer so that, as it moves on to the
            // next layer, the "previous layer" is already in hand instead of being decoded a second time.
            Parallel.For(0, SlicerFile.LayerCount, CoreSettings.ParallelOptions,
                () => new DecodedLayerCache(),
                (layerIndexInt, _, decodedLayers) =>
            {
                progress.PauseIfRequested();
                if (progress.Token.IsCancellationRequested)
                {
                    return decodedLayers;
                }

                uint layerIndex = (uint)layerIndexInt;
                var layer = SlicerFile[layerIndex];

                if (layer.IsEmpty)
                {
                    progress.LockAndIncrement();
                    return decodedLayers;
                }

                var abovePlate = layerIndex > 0 && layer.PositionZ > firstLayer!.PositionZ;
                var checkOverhangs = abovePlate && overhangConfig.Enabled &&
                                     (overhangConfig.WhiteListLayers is null || overhangConfig.WhiteListLayers.Contains(layerIndex));
                var checkIslands = abovePlate && islandConfig.Enabled &&
                                   (islandConfig.WhiteListLayers is null || islandConfig.WhiteListLayers.Contains(layerIndex));
                if (!touchBoundConfig.Enabled && !resinTrapConfig.Enabled && !checkOverhangs && !checkIslands)
                {
                    progress.LockAndIncrement();
                    return decodedLayers;
                }

                // Detectors scan the decoded bounding box directly; no full-size frame is materialized.
                var current = decodedLayers.Take(layerIndex) ?? LayerRunImage.FromLayer(layer);
                var frameSize = SlicerFile.Resolution;
                var frameRect = new Rectangle(Point.Empty, frameSize);

                if (touchBoundConfig.Enabled)
                {
                    // TouchingBounds Checker
                    List<Point> pixels = [];
                    bool touchTop = layer.BoundingRectangle.Top <= touchBoundConfig.MarginTop;
                    bool touchBottom = layer.BoundingRectangle.Bottom >= frameSize.Height - touchBoundConfig.MarginBottom;
                    bool touchLeft = layer.BoundingRectangle.Left <= touchBoundConfig.MarginLeft;
                    bool touchRight = layer.BoundingRectangle.Right >= frameSize.Width - touchBoundConfig.MarginRight;

                    int minx = int.MaxValue;
                    int miny = int.MaxValue;
                    int maxx = 0;
                    int maxy = 0;

                    // Pixels at or above the minimum brightness, frame coordinates, built only if a margin is touched
                    IntervalMap? bright = null;
                    bool Lit(int x, int y)
                    {
                        if (touchBoundConfig.MinimumPixelBrightness == 0) return true;
                        bright ??= current.Intervals((byte)(touchBoundConfig.MinimumPixelBrightness - 1), frameRect);
                        return bright.Contains(y, x);
                    }

                    void Touch(int x, int y)
                    {
                        pixels.Add(new Point(x, y));
                        minx = Math.Min(minx, x);
                        miny = Math.Min(miny, y);
                        maxx = Math.Max(maxx, x);
                        maxy = Math.Max(maxy, y);
                    }

                    if (touchTop || touchBottom)
                    {
                        for (int x = layer.BoundingRectangle.X; x < layer.BoundingRectangle.Right; x++) // Check Top and Bottom bounds
                        {
                            if (touchTop)
                            {
                                for (int y = layer.BoundingRectangle.Y; y < touchBoundConfig.MarginTop; y++) // Top
                                {
                                    if (Lit(x, y)) Touch(x, y);
                                }
                            }

                            if (touchBottom)
                            {
                                for (int y = frameSize.Height - touchBoundConfig.MarginBottom; y < layer.BoundingRectangle.Bottom; y++) // Bottom
                                {
                                    if (Lit(x, y)) Touch(x, y);
                                }
                            }
                        }
                    }

                    if (touchLeft || touchRight)
                    {
                        for (int y = layer.BoundingRectangle.Y + touchBoundConfig.MarginTop;
                             y < layer.BoundingRectangle.Bottom - touchBoundConfig.MarginBottom;
                             y++) // Check Left and Right bounds
                        {
                            if (touchLeft)
                            {
                                for (int x = layer.BoundingRectangle.X; x < touchBoundConfig.MarginLeft; x++) // Left
                                {
                                    if (Lit(x, y)) Touch(x, y);
                                }
                            }

                            if (touchRight)
                            {
                                for (int x = layer.BoundingRectangle.Right - touchBoundConfig.MarginRight; x < layer.BoundingRectangle.Right; x++) // Right
                                {
                                    if (Lit(x, y)) Touch(x, y);
                                }
                            }
                        }
                    }

                    if (pixels.Count > 0)
                    {
                        AddIssue(new MainIssue(MainIssue.IssueType.TouchingBound, new IssueOfPoints(layer, pixels,
                            new Rectangle(minx, miny, maxx - minx + 1, maxy - miny + 1))));
                    }
                }

                if (checkOverhangs || checkIslands)
                {
                    var previousLayer = SlicerFile[layerIndex - 1];
                    LayerRunImage? previous = null;
                    // Everything here is expressed in the union of both layers' rectangles, as it always was
                    var window = Layer.GetBoundingRectangleUnion(previousLayer, layer);
                    RunLengthImage? overhangRuns = null;
                    IntervalMap? overhangMap = null;

                    List<MainIssue>? overhangs = overhangConfig.Enabled ? [] : null;
                    if (checkOverhangs)
                    {
                        previous ??= decodedLayers.GetOrDecode(previousLayer);

                        // Compute and erode the current-minus-previous mask as runs. Only the surviving
                        // bounds are rasterized for contour tracing.
                        overhangRuns = current.Overhang(previous, window).ErodeCross(overhangsRadius);
                        var overhangBounds = overhangRuns.Bounds;
                        if (!overhangBounds.IsEmpty)
                        {
                            // One pixel of zero margin so that tracing sees a border, as it did on the full image
                            using var raster = decodedLayers.Scratch(new Size(overhangBounds.Width + 2, overhangBounds.Height + 2), 1);
                            overhangRuns.DrawInto(raster, new Point(1 - overhangBounds.X, 1 - overhangBounds.Y));
                            var contourOffset = new Point(window.X + overhangBounds.X - 1, window.Y + overhangBounds.Y - 1);

                            using var contours = raster.FindContours(out var hierarchy, RetrType.Tree, ChainApproxMethod.ChainApproxSimple, contourOffset);
                            var contoursInGroups = EmguContours.GetPositiveContoursInGroups(contours, hierarchy);

                            foreach (var contourGroup in contoursInGroups)
                            {
                                if (contourGroup[0].Size < 3) continue; // Single contour, single line, ignore
                                var area = EmguContours.GetContourArea(contourGroup);
                                if (area >= overhangConfig.RequiredPixelsToConsider)
                                {
                                    var rect = CvInvoke.BoundingRectangle(contourGroup[0]);
                                    var overhangIssue = new MainIssue(MainIssue.IssueType.Overhang, new IssueOfContours(layer, contourGroup.ToArrayOfArray(), rect, area));
                                    overhangs!.Add(overhangIssue);
                                    AddIssue(overhangIssue);
                                }
                            }
                        }
                    }

                    if (checkIslands)
                    {
                        // Foreground is every pixel above the binary threshold, the same set that
                        // Threshold(THRESH_BINARY) + ConnectedComponents used to label
                        var islandRuns = current.Runs(islandConfig.BinaryThreshold, window);
                        var components = islandRuns.LabelComponents(islandConfig.AllowDiagonalBonds);

                        // Pixels bright enough to count, and pixels of the previous layer bright enough to
                        // support; built when the first component needs them. A brightness of 0 means every pixel.
                        IntervalMap? brightMap = null;
                        IntervalMap? supportMap = null;

                        for (int i = 0; i < components.Count; i++)
                        {
                            if (components.Area(i) < islandConfig.RequiredAreaToProcessCheck) continue;

                            var rect = components.Bounds(i);
                            var componentRuns = components.RunsOf(i);

                            previous ??= decodedLayers.GetOrDecode(previousLayer);
                            brightMap ??= islandConfig.RequiredPixelBrightnessToProcessCheck == 0
                                ? IntervalMap.Full(window.Width, window.Height)
                                : current.Intervals((byte)(islandConfig.RequiredPixelBrightnessToProcessCheck - 1), window);
                            supportMap ??= islandConfig.RequiredPixelBrightnessToSupport == 0
                                ? IntervalMap.Full(window.Width, window.Height)
                                : previous.Intervals((byte)(islandConfig.RequiredPixelBrightnessToSupport - 1), window);

                            // First pass only counts. The point list is materialized later, and only
                            // for components that actually turn out to be islands: a large solid
                            // cross-section would otherwise grow (and immediately discard) a list with
                            // one entry per pixel.
                            long pixelCount = 0;
                            long pixelsSupportingIsland = 0;

                            foreach (var runIndex in componentRuns)
                            {
                                var run = components.GetRun(runIndex);
                                var brightRow = brightMap.Row(run.Y);
                                for (var b = 0; b < brightRow.Length; b += 2)
                                {
                                    if (brightRow[b] >= run.End) break;
                                    var lo = Math.Max(brightRow[b], run.Start);
                                    var hi = Math.Min(brightRow[b + 1], run.End);
                                    if (hi <= lo) continue;
                                    pixelCount += hi - lo;
                                    pixelsSupportingIsland += supportMap.IntersectionRow(run.Y, lo, hi);
                                }
                            }

                            if (pixelCount == 0) continue; // Should never happen

                            var requiredSupportingPixels = Math.Max(1, pixelCount * islandConfig.RequiredPixelsToSupportMultiplier);

                            if (pixelsSupportingIsland >= requiredSupportingPixels) continue;

                            var islandBoundingRectangle = rect.OffsetBy(window.Location);

                            // Check for overhangs in islands
                            if (islandConfig.EnhancedDetection && pixelsSupportingIsland >= 10 && pixelsSupportingIsland >= requiredSupportingPixels / 4)
                            {
                                if (overhangConfig.Enabled &&
                                    overhangs!.TrueForAll(overhang => !overhang.BoundingRectangle.IntersectsWith(islandBoundingRectangle)))
                                {
                                    continue;
                                }

                                // Overhang pixels inside this island's rectangle: from the layer's overhang mask
                                // when the overhang pass ran, otherwise computed for the rectangle alone.
                                long overhangPixels = 0;

                                if (overhangRuns is not null)
                                {
                                    overhangMap ??= IntervalMap.FromImage(overhangRuns);
                                    foreach (var runIndex in componentRuns)
                                    {
                                        if (overhangPixels >= overhangConfig.RequiredPixelsToConsider) break;
                                        var run = components.GetRun(runIndex);
                                        overhangPixels += overhangMap.IntersectionRow(run.Y, run.Start, run.End);
                                    }
                                }
                                else
                                {
                                    // No overhang pass this layer: decode just this rectangle of both layers
                                    using var currentRect = current.DecompressRectangle(islandBoundingRectangle);
                                    using var previousRect = previous.DecompressRectangle(islandBoundingRectangle);
                                    using var subtractedImage = IslandOverhangImage(currentRect, previousRect, new Rectangle(Point.Empty, rect.Size));
                                    var subtractedSpan = subtractedImage.GetReadOnlySpan2DOfBytes();

                                    // Only the component's own pixels count, so walk its runs; the subtracted image is relative to rect
                                    foreach (var runIndex in componentRuns)
                                    {
                                        if (overhangPixels >= overhangConfig.RequiredPixelsToConsider) break;
                                        var run = components.GetRun(runIndex);
                                        var subtractedRow = subtractedSpan.GetRowSpan(run.Y - rect.Y);
                                        for (int x = run.Start; x < run.End && overhangPixels < overhangConfig.RequiredPixelsToConsider; x++)
                                        {
                                            if (subtractedRow[x - rect.X] != 0) overhangPixels++;
                                        }
                                    }
                                }

                                if (overhangPixels < overhangConfig.RequiredPixelsToConsider) // No overhang = no island
                                {
                                    continue;
                                }
                            }

                            // Confirmed island: now collect its pixels, in raster order
                            var points = new List<Point>((int)pixelCount);
                            foreach (var runIndex in componentRuns)
                            {
                                var run = components.GetRun(runIndex);
                                var brightRow = brightMap.Row(run.Y);
                                for (var b = 0; b < brightRow.Length; b += 2)
                                {
                                    if (brightRow[b] >= run.End) break;
                                    var lo = Math.Max(brightRow[b], run.Start);
                                    var hi = Math.Min(brightRow[b + 1], run.End);
                                    for (int x = lo; x < hi; x++)
                                    {
                                        points.Add(new Point(window.X + x, window.Y + run.Y));
                                    }
                                }
                            }

                            AddIssue(new MainIssue(MainIssue.IssueType.Island, new IssueOfPoints(layer, points, islandBoundingRectangle)));
                        }

                    }
                }

                if (resinTrapConfig.Enabled)
                {
                    // Contours and run images are expressed in the file's bounding rectangle, the union of
                    // every layer, so the passes can fold layers into one air map. The work itself only
                    // needs this layer's own rectangle, with one pixel of margin so that contour tracing
                    // sees the same zero border it would see on the larger image.
                    var fileBounds = SlicerFile.BoundingRectangle;
                    var crop = layer.BoundingRectangle;
                    crop.Inflate(1, 1);
                    crop.Intersect(frameRect);
                    var cropOffset = new Point(crop.X - fileBounds.X, crop.Y - fileBounds.Y);
                    // The margin may poke outside the file rectangle when the layer touches its edge; the
                    // run images live inside that rectangle, and the margin is background anyway
                    var scanRect = Rectangle.Intersect(crop, fileBounds);
                    var scanOffset = new Point(scanRect.X - fileBounds.X, scanRect.Y - fileBounds.Y);
                    var scanInCrop = new Rectangle(scanRect.X - crop.X, scanRect.Y - crop.Y, scanRect.Width, scanRect.Height);

                    // Trace only this layer's padded bounds; the worker scratch buffer is already zeroed.
                    using (var contourLayer = decodedLayers.Scratch(crop.Size))
                    {
                        current.ThresholdInto(contourLayer, resinTrapConfig.BinaryThreshold, crop);
                        using var contours = contourLayer.FindContours(out var hierarchy, RetrType.Tree,
                            ChainApproxMethod.ChainApproxSimple, cropOffset);
                        externalContours[layerIndex] = EmguContours.GetExternalContours(contours, hierarchy);
                        hollows[layerIndex] = EmguContours.GetNegativeContoursInGroups(contours, hierarchy);
                        resinTrapsContoursArea[layerIndex] = EmguContours.GetContoursArea(hollows[layerIndex]);
                    }

                    // The passes used to threshold, invert and fill every layer again, twice, one layer at
                    // a time. Both maps they need are produced here instead while every worker
                    // is busy: the solid at the drain threshold, and the air outside the part, which is
                    // everything that is neither solid nor inside an outer contour.
                    var solid = current.Intervals(resinTrapConfig.MaximumPixelBrightnessToDrain, fileBounds);
                    solidMaps[layerIndex] = solid;
                    var layerAir = solid.Complement();
                    if (externalContours[layerIndex] is { Size: > 0 } externals)
                    {
                        using var outerFill = decodedLayers.Scratch(crop.Size);
                        CvInvoke.DrawContours(outerFill, externals, -1, EmguCvExtensions.WhiteColor, -1,
                            LineType.EightConnected, null, int.MaxValue, new Point(-cropOffset.X, -cropOffset.Y));
                        using var outerFillScan = new Mat(outerFill, scanInCrop);
                        layerAir.Subtract(IntervalMap.FromThreshold(outerFillScan.GetReadOnlySpan2DOfBytes(), 0, scanOffset, fileBounds.Size));
                    }

                    layerAirMaps[layerIndex] = layerAir;
                }

                decodedLayers.Keep(layerIndex, current);
                progress.LockAndIncrement();
                return decodedLayers;
            }, decodedLayers => decodedLayers.Dispose()); // Parallel end
        }

        if (progress.Token.IsCancellationRequested) return GetResult();

        if (resinTrapConfig.Enabled)
        {
            progress.Reset("Detection pass 1 of 2 (Resin traps)", SlicerFile.LayerCount, resinTrapConfig.StartLayerIndex);

            // The air map is folded layer by layer with the interval maps from the parallel pass, so the two
            // passes below never decode, threshold or rasterize a whole layer
            var fileBounds = SlicerFile.BoundingRectangle;
            IntervalMap? currentAirMap = null;

            // Folds one layer into the air map: air keeps whatever is not solid here, plus this layer's own
            // outside air. A layer without maps is an empty layer, which is all air.
            void FoldLayerIntoAirMap(IntervalMap? solid, IntervalMap? layerAir)
            {
                currentAirMap ??= layerAir is null ? IntervalMap.Full(fileBounds.Width, fileBounds.Height) : layerAir.Clone();
                if (layerAir is null) currentAirMap.SetAll();
                else if (solid is null) currentAirMap.Union(layerAir);
                // layerAir was formed from the complement of solid, so they cannot overlap. That makes
                // (air ∖ solid) ∪ layerAir equal to (air ∪ layerAir) ∖ solid, which has a one-sweep implementation.
                else currentAirMap.UnionThenSubtract(layerAir, solid);
            }

            /* the first pass does bottom to top, and tracks anything it thinks is a resin trap */
            for (var layerIndex = resinTrapConfig.StartLayerIndex; layerIndex < SlicerFile.LayerCount; layerIndex++)
            {
                if (progress.Token.IsCancellationRequested) return GetResult();

                FoldLayerIntoAirMap(solidMaps[layerIndex], layerAirMaps[layerIndex]);

                if (hollows[layerIndex] is not null)
                {
                    resinTraps[layerIndex] = [];
                    airContours[layerIndex] = [];

                    /* Phase 1 (parallel, read-only): rasterize every hollow and count how much known air it
                     * overlaps. Nothing is written here, so the answer does not depend on which thread gets
                     * to which hollow first. */
                    var hollowCount = hollows[layerIndex].Count;
                    var checks = new HollowAirCheck[hollowCount];
                    Parallel.For(0, hollowCount, CoreSettings.ParallelOptions, i =>
                    {
                        progress.PauseIfRequested();
                        if (progress.Token.IsCancellationRequested) return;
                        if (resinTrapsContoursArea[layerIndex][i] < resinTrapConfig.RequiredAreaToProcessCheck) return;
                        checks[i] = CheckHollowAgainstAirMap(hollows[layerIndex][i], currentAirMap!);
                    });

                    /* Phase 2 (sequential, in hollow order): classify each hollow and fold it into the air map.
                     * Applying the writes in a fixed order is what makes the result reproducible. */
                    for (var i = 0; i < hollowCount; i++)
                    {
                        var check = checks[i];
                        if (check.Fill is null || progress.Token.IsCancellationRequested) continue;

                        if (check.OverlapCount == 0)
                        {
                            /* this contour does *not* overlap known air */

                            /* add a resin trap (for now... will be revisited in part 2) */
                            resinTraps[layerIndex].Add(hollows[layerIndex][i]);
                        }
                        else if (check.OverlapCount >= resinTrapConfig.RequiredBlackPixelsToDrain)
                        {
                            /* this contour does overlap air, add it to the current air map and remember this contour was air-connected for 2nd pass */
                            airContours[layerIndex].Add(hollows[layerIndex][i]);

                            currentAirMap.Union(check.Fill, check.Roi.Location);
                        }
                        else
                        {
                            /* it overlapped ,but not by enough, treat as solid */
                            currentAirMap.Subtract(check.Fill, check.Roi.Location);
                        }
                    }
                }

                progress++;
            }

            if (progress.Token.IsCancellationRequested) return GetResult();
            progress.Reset("Detection pass 2 of 2 (Resin traps)", SlicerFile.LayerCount,
                resinTrapConfig.StartLayerIndex);
            /* starting over again but this time from the top to the bottom */
            currentAirMap = null;

            var resinTrapGroups = new List<List<(VectorOfVectorOfPoint contour, uint layerIndex)>>();
            var overlappingGroupIndexes = new List<int>();

            for (int layerIndex = resinTraps.Length - 1; layerIndex >= resinTrapConfig.StartLayerIndex; layerIndex--)
            {
                if (progress.Token.IsCancellationRequested) return GetResult();

                if (layerIndex == resinTraps.Length - 1)
                {
                    /* this is subtly different than the first pass: the initial air map is the inverse of the top layer, so anything open on the top layer is treated as air */
                    var topSolid = solidMaps[layerIndex];
                    currentAirMap = topSolid is null ? IntervalMap.Full(fileBounds.Width, fileBounds.Height) : topSolid.Complement();
                }

                FoldLayerIntoAirMap(solidMaps[layerIndex], layerAirMaps[layerIndex]);

                /* Update air map with any hollows that were found to be air-connected during first pass */
                if (airContours[layerIndex] is { Count: > 0 } airGroups)
                {
                    var airFills = new HollowAirCheck[airGroups.Count];
                    Parallel.For(0, airGroups.Count, CoreSettings.ParallelOptions, i =>
                    {
                        progress.PauseIfRequested();
                        airFills[i] = RasterizeGroup(airGroups[i], fileBounds.Size);
                    });

                    foreach (var airFill in airFills)
                    {
                        if (airFill.Fill is not null) currentAirMap!.Union(airFill.Fill, airFill.Roi.Location);
                    }
                }

                if (resinTraps[layerIndex] is not null)
                {
                    suctionCups[layerIndex] = [];
                    /* here we don't worry about finding contours on the layer, the bottom to top pass did that already */
                    /* all we care about is contours the first pass thought were resin traps, since there was no access to air from the bottom */

                    /* Phase 1 (parallel, read-only): does each candidate overlap known air, confined to its bounding box */
                    var trapCount = resinTraps[layerIndex].Count;
                    var checks = new HollowAirCheck[trapCount];
                    Parallel.For(0, trapCount, CoreSettings.ParallelOptions, x =>
                    {
                        progress.PauseIfRequested();
                        if (progress.Token.IsCancellationRequested) return;
                        checks[x] = CheckHollowAgainstAirMap(resinTraps[layerIndex][x], currentAirMap!);
                    });

                    /* Phase 2 (sequential, in candidate order): update the air map and the trap groups.
                     * The group bookkeeping below is order sensitive, so it must not run in thread-arrival order. */
                    for (var x = 0; x < trapCount; x++)
                    {
                        var check = checks[x];
                        if (check.Fill is null || progress.Token.IsCancellationRequested) continue;
                        var trap = resinTraps[layerIndex][x];

                        if (check.OverlapCount >= resinTrapConfig.RequiredBlackPixelsToDrain)
                        {
                            /* this contour does overlap air, add this it our air map */
                            currentAirMap!.Union(check.Fill, check.Roi.Location);
                            /* Always add the removed contour to suctionTraps (even if we aren't reporting suction traps)
                             * This is because contours that are placed on here get removed from resin traps in the next stage
                             * if you don't put them here, they never get removed even if they should :) */

                            /* since we know it isn't a resin trap, it becomes a suction trap */
                            suctionCups[layerIndex].Add(trap);

                            for (var groupIndex = resinTrapGroups.Count - 1; groupIndex >= 0; groupIndex--)
                            {
                                var group = resinTrapGroups[groupIndex];
                                if (group[^1].layerIndex > layerIndex + 1)
                                {
                                    // this group is disconnected from current layer by at least 1 layer, no need to process anything from here anymore
                                    continue;
                                }

                                // if any contours in this group, that are on the previous layer, overlap the new suction area, they are all suction areas
                                if (!GroupTouches(group, trap, layerIndex)) continue;

                                foreach (var item in group)
                                {
                                    suctionCups[item.layerIndex].Add(item.contour);
                                    if (item.layerIndex != layerIndex)
                                    {
                                        resinTraps[item.layerIndex].Remove(item.contour);
                                    }
                                }

                                group.Clear();
                                resinTrapGroups.RemoveAt(groupIndex);
                            }
                            /* to keep things tidy while we iterate resin traps, it will be left in the list for now, and removed later */
                        }
                        else
                        {
                            /* doesn't overlap by enough, remove from air map */
                            currentAirMap!.Subtract(check.Fill, check.Roi.Location);

                            /* put it in a group of resin traps, used when a subsequent layer becomes a suction cup, it can convert any overlapping groups to suction cup */
                            overlappingGroupIndexes.Clear();
                            for (var groupIndex = 0; groupIndex < resinTrapGroups.Count; groupIndex++)
                            {
                                /* the last entry is always the lowest layer seen so far, so this group is already out of reach */
                                if (resinTrapGroups[groupIndex][^1].layerIndex > layerIndex + 1) continue;

                                if (GroupTouches(resinTrapGroups[groupIndex], trap, layerIndex))
                                {
                                    overlappingGroupIndexes.Add(groupIndex);
                                }
                            }

                            if (overlappingGroupIndexes.Count == 0)
                            {
                                // no overlaps, make a single issue
                                resinTrapGroups.Add([(trap, (uint)layerIndex)]);
                            }
                            else if (overlappingGroupIndexes.Count == 1)
                            {
                                resinTrapGroups[overlappingGroupIndexes[0]].Add((trap, (uint)layerIndex));
                            }
                            else
                            {
                                var combinedGroup = new List<(VectorOfVectorOfPoint contour, uint layerIndex)>();
                                foreach (var index in overlappingGroupIndexes)
                                {
                                    combinedGroup.AddRange(resinTrapGroups[index]);
                                }

                                for (var index = overlappingGroupIndexes.Count - 1; index >= 0; index--)
                                {
                                    resinTrapGroups[overlappingGroupIndexes[index]].Clear();
                                    resinTrapGroups.RemoveAt(overlappingGroupIndexes[index]);
                                }

                                combinedGroup.Add((trap, (uint)layerIndex));
                                resinTrapGroups.Add(combinedGroup);
                            }
                        }
                    }

                    /* anything that converted to a suction trap needs to removed from resinTraps. Loop backwards so indexes don't shift */
                    if (suctionCups[layerIndex] is not null)
                    {
                        for (var i = suctionCups[layerIndex].Count - 1; i >= 0; i--)
                        {
                            resinTraps[layerIndex].Remove(suctionCups[layerIndex][i]);
                            if (resinTraps[layerIndex].Count > 0) continue;
                            resinTraps[layerIndex] = null!;
                            break;
                        }
                    }
                }

                progress++;
            }

            if (progress.Token.IsCancellationRequested) return GetResult();

            /* translate all contour points by ROI x and y */
            var offsetBy = new Point(SlicerFile.BoundingRectangle.X, SlicerFile.BoundingRectangle.Y);
            foreach (var listOfLayers in new[] { resinTraps, suctionCups })
            {
                Parallel.ForEach(listOfLayers.Where(list => list is not null), contoursGroups =>
                {
                    progress.PauseIfRequested();
                    for (var groupIndex = 0; groupIndex < contoursGroups.Count; groupIndex++)
                    {
                        var contours = contoursGroups[groupIndex];

                        var arrayOfArrayOfPoints = contours.ToArrayOfArray();

                        foreach (var pointArray in arrayOfArrayOfPoints)
                            for (var i = 0; i < pointArray.Length; i++)
                                pointArray[i].Offset(offsetBy);

                        contoursGroups[groupIndex].Dispose();
                        contoursGroups[groupIndex] = new VectorOfVectorOfPoint(arrayOfArrayOfPoints);
                    }

                });
            }

            if (progress.Token.IsCancellationRequested) return GetResult();

            if (resinTrapConfig.DetectSuctionCups)
                progress.Reset("Interpolating areas (Resin traps & suction cups)",
                    (uint)(resinTraps.Count(list => list is not null) + suctionCups.Count(list => list is not null)));
            else
                progress.Reset("Interpolating areas (Resin traps)", (uint)(resinTraps.Count(list => list is not null)));

            Parallel.Invoke(() =>
                {
                    var resinTrapGroups = new List<List<IssueOfContours>>();
                    var overlappingGroupIndexes = new List<int>();

                    for (var layerIndex = resinTraps.Length - 1; layerIndex >= 0; layerIndex--)
                    {
                        if (resinTraps[layerIndex] is null) continue;

                        /* select new LayerIssue(this[layerIndex], LayerIssue.IssueType.ResinTrap, area.Contour, area.BoundingRectangle)) */
                        foreach (var trap in resinTraps[layerIndex])
                        {
                            progress.PauseIfRequested();
                            if (progress.Token.IsCancellationRequested) return;

                            var area = EmguContours.GetContourArea(trap);
                            var rect = CvInvoke.BoundingRectangle(trap[0]);
                            var trapIssue = new IssueOfContours(SlicerFile[layerIndex], trap.ToArrayOfArray(), rect,
                                area);

                            overlappingGroupIndexes.Clear();
                            for (var x = 0; x < resinTrapGroups.Count; x++)
                            {
                                if (resinTrapGroups[x][^1].LayerIndex > layerIndex + 1) continue;

                                if (IssueGroupTouches(resinTrapGroups[x], trap, rect, layerIndex))
                                {
                                    overlappingGroupIndexes.Add(x);
                                }
                            }

                            if (overlappingGroupIndexes.Count == 0)
                            {
                                /* no overlaps, make a single issue */
                                resinTrapGroups.Add([trapIssue]);
                            }
                            else if (overlappingGroupIndexes.Count == 1)
                            {
                                resinTrapGroups[overlappingGroupIndexes[0]].Add(trapIssue);
                            }
                            else
                            {
                                var combinedGroup = new List<IssueOfContours>();
                                foreach (var index in overlappingGroupIndexes)
                                {
                                    combinedGroup.AddRange(resinTrapGroups[index]);
                                }

                                for (var index = overlappingGroupIndexes.Count - 1; index >= 0; index--)
                                {
                                    resinTrapGroups[overlappingGroupIndexes[index]].Clear();
                                    resinTrapGroups.RemoveAt(overlappingGroupIndexes[index]);
                                }

                                combinedGroup.Add(trapIssue);
                                resinTrapGroups.Add(combinedGroup);
                            }
                        }

                        progress.LockAndIncrement();
                    }

                    foreach (var group in resinTrapGroups)
                    {
                        if (group.AsValueEnumerable().Any(issue => issue.LayerIndex == 0)) continue; // Not a trap if on plate
                        AddIssue(new MainIssue(MainIssue.IssueType.ResinTrap, group));
                    }
                },
                () =>
                {
                    /* only report suction cup issues if enabled */
                    if (resinTrapConfig.DetectSuctionCups)
                    {
                        var minimumSuctionArea = resinTrapConfig.RequiredAreaToConsiderSuctionCup;
                        var suctionGroups = new List<List<IssueOfContours>>();
                        var overlappingGroupIndexes = new List<int>();

                        for (var layerIndex = suctionCups.Length - 1; layerIndex >= 0; layerIndex--)
                        {
                            if (suctionCups[layerIndex] is null) continue;

                            foreach (var trap in suctionCups[layerIndex])
                            {
                                progress.PauseIfRequested();
                                if (progress.Token.IsCancellationRequested) return;

                                var area = EmguContours.GetContourArea(trap);
                                if (area < minimumSuctionArea) continue;
                                var rect = CvInvoke.BoundingRectangle(trap[0]);

                                var trapIssue = new IssueOfContours(SlicerFile[layerIndex], trap.ToArrayOfArray(), rect,
                                    area);

                                overlappingGroupIndexes.Clear();
                                for (var x = 0; x < suctionGroups.Count; x++)
                                {
                                    if (suctionGroups[x][^1].LayerIndex > layerIndex + 1) continue;
                                    if (IssueGroupTouches(suctionGroups[x], trap, rect, layerIndex))
                                    {
                                        overlappingGroupIndexes.Add(x);
                                    }
                                }

                                if (overlappingGroupIndexes.Count == 0)
                                {
                                    /* no overlaps, make a new group */
                                    suctionGroups.Add([trapIssue]);
                                }
                                else if (overlappingGroupIndexes.Count == 1)
                                {
                                    suctionGroups[overlappingGroupIndexes[0]].Add(trapIssue);
                                }
                                else
                                {
                                    var combinedGroup = new List<IssueOfContours>();
                                    /* iterate backwards to not screw up indexes */
                                    for (var i = overlappingGroupIndexes.Count - 1; i >= 0; i--)
                                    {
                                        var index = overlappingGroupIndexes[i];
                                        combinedGroup.AddRange(suctionGroups[index]);
                                        suctionGroups[index].Clear();
                                        suctionGroups.RemoveAt(index);
                                    }

                                    combinedGroup.Add(trapIssue);
                                    suctionGroups.Add(combinedGroup);
                                }
                            }

                            progress.LockAndIncrement();
                        }

                        foreach (var group in suctionGroups)
                        {
                            var mainIssue = new MainIssue(MainIssue.IssueType.SuctionCup, group);
                            if ((decimal)mainIssue.TotalHeight >= resinTrapConfig.RequiredHeightToConsiderSuctionCup)
                            {
                                AddIssue(mainIssue);
                            }
                        }
                    }
                });

            // Dispose
            foreach (var listOfVectors in new[] { resinTraps, suctionCups, hollows, airContours })
            {
                foreach (var vectorArray in listOfVectors)
                {
                    if (vectorArray is null) continue;
                    foreach (var vector in vectorArray)
                    {
                        vector?.Dispose();
                    }
                }
            }

            foreach (var vector in externalContours)
            {
                vector?.Dispose();
            }
        }

        return GetResult();
    }

    public MainIssue[] DrillSuctionCupsForIssues(IEnumerable<MainIssue> issues, int ventHoleDiameter,
        OperationProgress progress)
    {
        var drillOps = new List<PixelOperation>();
        var drilledIssues = new List<MainIssue>();
        var radius = SlicerFile.PixelsToNormalizedPitch(ventHoleDiameter / 2);
        /* for each suction cup issue that is an initial layer */
        foreach (var mainIssue in issues)
        {
            var drillPoint = GetDrillLocation((IssueOfContours)mainIssue[0], radius);
            if (drillPoint.IsAnyNegative()) continue;
            drillOps.Add(new PixelDrainHole(mainIssue.StartLayerIndex, drillPoint, (ushort)ventHoleDiameter));
            drilledIssues.Add(mainIssue);
        }

        SlicerFile.DrawModifications(drillOps, progress);

        return drilledIssues.ToArray();
    }

    public static Point GetDrillLocation(IssueOfContours issue, Size radius)
    {
        using var vecCentroid = new VectorOfPoint(issue.Contours[0]);
        var centroid = EmguContour.GetCentroid(vecCentroid);
        if (centroid.IsAnyNegative()) return centroid;
        using var circleCheck = EmguCvExtensions.InitMat(issue.BoundingRectangle.Size);
        using var contourMat = EmguCvExtensions.InitMat(issue.BoundingRectangle.Size);

        var inverseOffset = new Point(issue.BoundingRectangle.X * -1, issue.BoundingRectangle.Y * -1);
        using var vec = new VectorOfVectorOfPoint(issue.Contours);
        CvInvoke.DrawContours(contourMat, vec, -1, EmguCvExtensions.WhiteColor, -1, LineType.EightConnected, null,
            int.MaxValue, inverseOffset);
        circleCheck.DrawCircle(new(centroid.X + inverseOffset.X, centroid.Y + inverseOffset.Y), radius,
            EmguCvExtensions.WhiteColor, -1);
        CvInvoke.BitwiseAnd(circleCheck, contourMat, circleCheck);

        return CvInvoke.HasNonZero(circleCheck)
            ? centroid       /* 5px centroid is inside layer! drill baby drill */
            : new Point(-1, -1); /* centroid is not inside the actual contour, no drill */
    }
}