using Microsoft.ML;
using CreditCardFraud_Model;

// Load the model zip that was copied next to the .exe
var mlContext = new MLContext();
var modelPath = Path.Combine(AppContext.BaseDirectory, "MLModel.zip");
ITransformer model = mlContext.Model.Load(modelPath, out _);

// Create a prediction engine with the generated input/output classes
var engine = mlContext.Model.CreatePredictionEngine<
    CreditCardFraudModel.ModelInput,
    CreditCardFraudModel.ModelOutput>(model);

// One hard-coded record for testing
var input = new CreditCardFraudModel.ModelInput
{
    TransactionAmount = 12.98F,
    HourOfDay = 1F,
    MerchantCategory = "Groceries",
    EntryMode = "Chip",
    Country = "US",
    IsInternational = 0F,
    CardPresent = 1F,
    DaysSinceLastTransaction = 1.65F,
    TransactionCount24h = 1F,
    AvgAmount7d = 11.91F
};

var result = engine.Predict(input);

Console.WriteLine("=== Prediction Input ===");
Console.WriteLine($"TransactionAmount:        {input.TransactionAmount}");
Console.WriteLine($"HourOfDay:                {input.HourOfDay}");
Console.WriteLine($"MerchantCategory:         {input.MerchantCategory}");
Console.WriteLine($"EntryMode:                {input.EntryMode}");
Console.WriteLine($"Country:                  {input.Country}");
Console.WriteLine($"IsInternational:          {input.IsInternational}");
Console.WriteLine($"CardPresent:              {input.CardPresent}");
Console.WriteLine($"DaysSinceLastTransaction: {input.DaysSinceLastTransaction}");
Console.WriteLine($"TransactionCount24h:      {input.TransactionCount24h}");
Console.WriteLine($"AvgAmount7d:              {input.AvgAmount7d}");

Console.WriteLine();
Console.WriteLine("=== Prediction Result ===");
Console.WriteLine($"Predicted Label: {(result.PredictedLabel ? "Fraud" : "Not Fraud")}");
Console.WriteLine($"Probability:     {result.Probability:P2}");