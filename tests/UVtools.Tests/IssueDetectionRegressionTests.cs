using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using UVtools.Core.FileFormats;
using UVtools.Core.Layers;
using UVtools.Core.Operations;
using Xunit;

namespace UVtools.Tests;

public class IssueDetectionRegressionTests
{
    /// <summary>
    /// Builds a file on the PCB fixture plate (1000px, 10 px/mm, 0.05mm layers) with one layer per flag:
    /// true draws a solid square, false leaves the layer empty.
    /// </summary>
    private static FileFormat CreateFile(params bool[] solidLayers)
    {
        var slicerFile = PcbFixtures.CreateSlicerFile();
        var layers = new Layer[solidLayers.Length];
        for (var i = 0; i < layers.Length; i++)
        {
            using var mat = new Mat(slicerFile.Resolution, DepthType.Cv8U, 1);
            mat.SetTo(new MCvScalar(0));
            if (solidLayers[i]) CvInvoke.Rectangle(mat, new Rectangle(400, 400, 200, 200), new MCvScalar(255), -1);
            layers[i] = new Layer((uint)i, mat, slicerFile);
        }

        slicerFile.Init(layers);
        return slicerFile;
    }

    private static uint[] EmptyLayerIssues(FileFormat slicerFile, bool ignoreStarting, bool ignoreLoose, bool ignoreEnding)
    {
        var config = new IssuesDetectionConfiguration();
        config.DisableAll();
        config.EmptyLayerConfig.Enabled = true;
        config.EmptyLayerConfig.IgnoreStartingEmptyLayers = ignoreStarting;
        config.EmptyLayerConfig.IgnoreLooseEmptyLayers = ignoreLoose;
        config.EmptyLayerConfig.IgnoreEndingEmptyLayers = ignoreEnding;

        // The classifier used to advance the outer layer index instead of its own cursor and never returned
        var detect = Task.Run(() => slicerFile.IssueManager.DetectIssues(config, new OperationProgress()));
        Assert.True(detect.Wait(TimeSpan.FromSeconds(30)), "empty layer detection did not finish");

        return detect.Result
            .Where(issue => issue.Type == MainIssue.IssueType.EmptyLayer)
            .Select(issue => issue.StartLayerIndex)
            .OrderBy(index => index)
            .ToArray();
    }

    [Fact]
    public void EmptyLayers_AreClassifiedAsStartingLooseOrEnding()
    {
        // 0,1 empty (starting) | 2 solid | 3 empty (loose) | 4 solid | 5,6 empty (ending)
        using var slicerFile = CreateFile(false, false, true, false, true, false, false);

        Assert.Equal(new uint[] { 0, 1, 3, 5, 6 }, EmptyLayerIssues(slicerFile, false, false, false));
        Assert.Equal(new uint[] { 3, 5, 6 }, EmptyLayerIssues(slicerFile, ignoreStarting: true, ignoreLoose: false, ignoreEnding: false));
        Assert.Equal(new uint[] { 0, 1, 5, 6 }, EmptyLayerIssues(slicerFile, ignoreStarting: false, ignoreLoose: true, ignoreEnding: false));
        Assert.Equal(new uint[] { 0, 1, 3 }, EmptyLayerIssues(slicerFile, ignoreStarting: false, ignoreLoose: false, ignoreEnding: true));
        Assert.Empty(EmptyLayerIssues(slicerFile, ignoreStarting: true, ignoreLoose: true, ignoreEnding: true));
    }

    [Fact]
    public void MainIssue_GroupTotalIsAnAreaOnOneLayerAndAVolumeAcrossLayers()
    {
        using var slicerFile = CreateFile(true, true, true);
        var square = new[] { new[] { new Point(0, 0), new Point(10, 0), new Point(10, 10), new Point(0, 10) } };
        IssueOfContours At(uint layerIndex, double area) =>
            new(slicerFile[layerIndex], square, new Rectangle(0, 0, 11, 11), area);

        var single = new MainIssue(MainIssue.IssueType.ResinTrap, new List<Issue> { At(1, 30) });
        Assert.Equal(30, single.Area, 3);
        Assert.Equal('²', single.AreaChar);

        var sameLayer = new MainIssue(MainIssue.IssueType.ResinTrap, new List<Issue> { At(1, 30), At(1, 12) });
        Assert.Equal(42, sameLayer.Area, 3);
        Assert.Equal('²', sameLayer.AreaChar);

        // Given out of order on purpose: children must come back sorted by layer
        var stacked = new MainIssue(MainIssue.IssueType.ResinTrap, new List<Issue> { At(2, 30), At(1, 12) });
        var expectedVolume = 30 * slicerFile.MillimetersToPixelsF(slicerFile[2].LayerHeight)
                             + 12 * slicerFile.MillimetersToPixelsF(slicerFile[1].LayerHeight);
        Assert.Equal(Math.Round(expectedVolume, 3), stacked.Area, 3);
        Assert.Equal('³', stacked.AreaChar);
        Assert.Equal(1u, stacked.StartLayerIndex);
        Assert.Equal(2u, stacked.EndLayerIndex);
        Assert.Equal(2, stacked.Count);
    }

    [Fact]
    public void FindByExtensionOrFilePath_ResolvesAPathThatDoesNotExistYet()
    {
        // .ctb is shared by the plain and the encrypted Chitubox formats. Nothing can inspect a file that
        // has not been written, so a conversion target must fall back to the plain variant instead of null.
        var path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName(), "not-written-yet.ctb");
        Assert.False(File.Exists(path));

        Assert.IsType<ChituboxFile>(FileFormat.FindByExtensionOrFilePath(path, true));
        Assert.IsType<ChituboxFile>(FileFormat.FindByExtensionOrFilePath("ctb", true));
        Assert.Null(FileFormat.FindByExtensionOrFilePath(Path.ChangeExtension(path, ".not-a-slicer-format"), true));
    }
}
