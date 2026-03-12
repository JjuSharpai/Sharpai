using Microsoft.ML;
using Microsoft.ML.Data;

var mlContext = new MLContext(seed: 42);

// 데이터 로드 (UCI Iris 데이터셋)
var dataPath = Path.Combine(AppContext.BaseDirectory, "iris.data");

if (!File.Exists(dataPath))
{
    using var client = new HttpClient();
    var csv = await client.GetStringAsync(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data");
    await File.WriteAllTextAsync(dataPath, csv);
    Console.WriteLine("Iris 데이터셋 다운로드 완료");
}

var dataView = mlContext.Data.LoadFromTextFile<IrisData>(
    dataPath, separatorChar: ',', hasHeader: false);

// 학습/테스트 분할 (70/30)
var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.3, seed: 42);

// 파이프라인: 피처 결합 → 라벨 매핑 → SVM(LinearSvm) 학습
var pipeline = mlContext.Transforms.Concatenate(
        "Features", nameof(IrisData.SepalLength), nameof(IrisData.SepalWidth),
        nameof(IrisData.PetalLength), nameof(IrisData.PetalWidth))
    .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
    .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(
        mlContext.BinaryClassification.Trainers.LinearSvm()))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

// 학습
Console.WriteLine("모델 학습 중...");
var model = pipeline.Fit(split.TrainSet);

// 추론 및 평가
var predictions = model.Transform(split.TestSet);
var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

Console.WriteLine($"\nAccuracy: {metrics.MacroAccuracy:F4}");
Console.WriteLine($"Log-Loss: {metrics.LogLoss:F4}");
Console.WriteLine($"\n{metrics.ConfusionMatrix.GetFormattedConfusionTable()}");

// 단일 샘플 추론 예시
var predEngine = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model);
var sample = new IrisData
{
    SepalLength = 5.1f, SepalWidth = 3.5f,
    PetalLength = 1.4f, PetalWidth = 0.2f
};
var result = predEngine.Predict(sample);
Console.WriteLine($"샘플 추론 결과: {result.PredictedLabel}");

// 입력 데이터 클래스
public class IrisData
{
    [LoadColumn(0)] public float SepalLength { get; set; }
    [LoadColumn(1)] public float SepalWidth { get; set; }
    [LoadColumn(2)] public float PetalLength { get; set; }
    [LoadColumn(3)] public float PetalWidth { get; set; }
    [LoadColumn(4)] public string Label { get; set; } = string.Empty;
}

// 추론 결과 클래스
public class IrisPrediction
{
    public string PredictedLabel { get; set; } = string.Empty;
}
