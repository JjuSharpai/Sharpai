#:project ../src/SharpAI.Domain/SharpAI.Domain.csproj
#:project ../src/SharpAI.Application/SharpAI.Application.csproj
#:project ../src/SharpAI.Infrastructure/SharpAI.Infrastructure.csproj

using SharpAI.Application.Services;
using SharpAI.Infrastructure.Data;
using SharpAI.Infrastructure.SupervisedLearning;

var labelMap = new Dictionary<string, double>
{
    ["Iris-setosa"] = 0,
    ["Iris-versicolor"] = 1,
    ["Iris-virginica"] = 2
};

var loader = new CsvDataLoader();
var dataSet = await loader.LoadAsync(
    url: "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    parser: parts => (
        Features: [double.Parse(parts[0]), double.Parse(parts[1]), double.Parse(parts[2]), double.Parse(parts[3])],
        Label: labelMap[parts[4].Trim()]
    ));

var (trainSet, testSet) = DataSetSplitter.Split(dataSet, testRatio: 0.3, seed: 42);

var learner = new SVM(learningRate: 0.001, lambda: 0.01, epochs: 1000);
var pipeline = new TrainingPipeline();
var result = pipeline.RunSupervised(learner, trainSet, testSet);

Console.WriteLine($"Accuracy: {result.Metrics!["accuracy"]:F4}");
Console.WriteLine($"\n예측 결과 (처음 10개):");

string[] classNames = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];
for (int i = 0; i < Math.Min(10, result.Predictions.Length); i++)
{
    var actual = classNames[(int)testSet.Labels![i]];
    var predicted = classNames[(int)result.Predictions[i]];
    var mark = actual == predicted ? "O" : "X";
    Console.WriteLine($"  [{mark}] 실제: {actual,-16} 예측: {predicted}");
}
