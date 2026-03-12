using SharpAI.Domain.Models;

namespace SharpAI.Application.Services;

public static class DataSetSplitter
{
    public static (DataSet Train, DataSet Test) Split(DataSet data, double testRatio, int seed = 42)
    {
        var (train, _, test) = Split(data, 1.0 - testRatio, 0, testRatio, seed);
        return (train, test);
    }

    public static (DataSet Train, DataSet Validation, DataSet Test) Split(
        DataSet data, double trainRatio, double validationRatio, double testRatio, int seed = 42)
    {
        var rng = new Random(seed);
        var indices = Enumerable.Range(0, data.Features.Length).OrderBy(_ => rng.Next()).ToArray();

        var total = trainRatio + validationRatio + testRatio;
        int trainCount = (int)(data.Features.Length * (trainRatio / total));
        int valCount = (int)(data.Features.Length * (validationRatio / total));

        var trainIdx = indices[..trainCount];
        var valIdx = indices[trainCount..(trainCount + valCount)];
        var testIdx = indices[(trainCount + valCount)..];

        return (
            BuildSubset(data, trainIdx),
            BuildSubset(data, valIdx),
            BuildSubset(data, testIdx)
        );
    }

    private static DataSet BuildSubset(DataSet data, int[] indices) =>
        new(indices.Select(i => data.Features[i]).ToArray(),
            data.Labels is not null ? indices.Select(i => data.Labels[i]).ToArray() : null);
}
