using CreditCardFraud_Model; // must match the namespace in consumption.cs / training.cs

internal class Program
{
    static void Main()
    {
        Console.WriteLine("=== Credit Card Fraud AutoML: Train --> Save ZIP ===");

        // Save the model to a predictable place under this project's output
        var outputDir = Path.Combine(AppContext.BaseDirectory, "TrainedModels");
        Directory.CreateDirectory(outputDir);

        var zipPath = Path.Combine(outputDir, "CreditCardFraudModel.zip");

        // CSV is stored in this project's root (three folders above bin\Debug\net10.0)
        var projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
        var csvPath = Path.Combine(projectRoot, "credit_card_fraud_1000.csv");

        Console.WriteLine($"CSV: {csvPath}");
        Console.WriteLine($"ZIP: {zipPath}");

        if (!File.Exists(csvPath))
        {
            Console.WriteLine("CSV not found. Put credit_card_fraud_1000.csv in the project root.");
            return;
        }

        // Run the AutoML-selected pipeline and save the model to the ZIP path
        CreditCardFraudModel.Train(outputModelPath: zipPath, inputDataFilePath: csvPath);

        Console.WriteLine("Credit Card Fraud model ZIP created.");
    }
}