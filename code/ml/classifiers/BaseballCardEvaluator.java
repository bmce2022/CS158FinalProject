package ml.classifiers;

import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Evaluator for baseball card price prediction using neural network
 */
public class BaseballCardEvaluator {
    
    /**
     * Train and evaluate the neural network on baseball card data
     * 
     * @param dataFile path to the preprocessed CSV file
     */
    public static void runExperiment(String dataFile) {
        // Load the dataset
        DataSet fullData = new DataSet(dataFile, DataSet.CSVFILE);
        
        // Split into 80% train, 20% test (randomized)
        DataSetSplit split = fullData.split(0.8);
        DataSet trainData = split.getTrain();
        DataSet testData = split.getTest();
        
        // Create and configure the neural network
        TwoLayerNN nn = new TwoLayerNN(10);
        nn.setEta(0.01);
        nn.setIterations(5000);
        nn.setNormalizationParams(108.49, 655.00);
        
        // Train the model
        nn.train(trainData);
        
        // Generate predictions table
        generatePredictionsTable(nn, testData);
        
        // Save to CSV
        savePredictionsToCSV(nn, testData, "/Users/bleeckercoyne/Desktop/CS 158/CS158FinalProject/data/predictions.csv");
    }
    
    /**
     * Generate and print predictions table
     * 
     * @param nn the trained neural network
     * @param data the test dataset
     */
    private static void generatePredictionsTable(TwoLayerNN nn, DataSet data) {
        double minPrice = 108.49;
        double maxPrice = 655.00;
        
        // Print header
        System.out.println("│ Example  │ Actual Price  │ Predicted Price │   Error    │ Percent Error │ Squared Error  │");
        
        // Track aggregate metrics
        double totalSquaredError = 0.0;
        double totalAbsoluteError = 0.0;
        double totalPercentError = 0.0;
        double maxAbsoluteError = 0.0;
        
        int count = 1;
        for (Example example : data.getData()) {
            double actualNormalized = example.getLabel();
            double predictedNormalized = nn.predictValue(example);
            
            double actualPrice = denormalize(actualNormalized, minPrice, maxPrice);
            double predictedPrice = denormalize(predictedNormalized, minPrice, maxPrice);
            double error = actualPrice - predictedPrice;
            double percentError = Math.abs(error / actualPrice * 100);
            double squaredError = error * error;
            
            System.out.printf("│ %8d │ $%11.2f │ $%14.2f │ $%9.2f │ %12.2f%% │ $%13.2f │%n", 
                count, actualPrice, predictedPrice, error, percentError, squaredError);
            
            // Accumulate metrics
            totalSquaredError += squaredError;
            totalAbsoluteError += Math.abs(error);
            totalPercentError += percentError;
            maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(error));
            
            count++;
        }
        
        // Calculate and display aggregate metrics
        int n = data.getData().size();
        double mse = totalSquaredError / n;
        double rmse = Math.sqrt(mse);
        double mae = totalAbsoluteError / n;
        double mape = totalPercentError / n;
        
        System.out.println("\n=== Performance Metrics ===");
        System.out.println("Mean Squared Error (MSE):           $" + String.format("%,12.2f", mse));
        System.out.println("Root Mean Squared Error (RMSE):     $" + String.format("%,12.2f", rmse));
        System.out.println("Mean Absolute Error (MAE):          $" + String.format("%,12.2f", mae));
        System.out.println("Mean Absolute Percent Error (MAPE): " + String.format("%13.2f%%", mape));
        System.out.println("Max Absolute Error:                 $" + String.format("%,12.2f", maxAbsoluteError));
        System.out.println();
    }
    
    /**
     * Save predictions to CSV file
     * 
     * @param nn the trained neural network
     * @param data the test dataset
     * @param outputFile path to output CSV file
     */
    private static void savePredictionsToCSV(TwoLayerNN nn, DataSet data, String outputFile) {
        double minPrice = 108.49;
        double maxPrice = 655.00;
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            // Write header
            writer.println("Example,Actual_Price,Predicted_Price,Error,Percent_Error,Squared_Error");
            
            int count = 1;
            for (Example example : data.getData()) {
                double actualNormalized = example.getLabel();
                double predictedNormalized = nn.predictValue(example);
                
                double actualPrice = denormalize(actualNormalized, minPrice, maxPrice);
                double predictedPrice = denormalize(predictedNormalized, minPrice, maxPrice);
                double error = actualPrice - predictedPrice;
                double percentError = Math.abs(error / actualPrice * 100);
                double squaredError = error * error;
                
                writer.printf("%d,%.2f,%.2f,%.2f,%.2f,%.2f%n", 
                    count, actualPrice, predictedPrice, error, percentError, squaredError);
                count++;
            }
            
            System.out.println("\nPredictions saved to: " + outputFile);
            
        } catch (IOException e) {
            System.err.println("Error writing to CSV file: " + e.getMessage());
        }
    }
    
    /**
     * Denormalize a value from [0,1] to actual price
     * 
     * @param normalized normalized value
     * @param min minimum price
     * @param max maximum price
     * @return actual price
     */
    private static double denormalize(double normalized, double min, double max) {
        double clamped = Math.max(0.0, Math.min(1.0, normalized));
        return clamped * (max - min) + min;
    }
    
    /**
     * Main method to run the experiment
     */
    public static void main(String[] args) {
        runExperiment("/Users/bleeckercoyne/Desktop/CS 158/CS158FinalProject/data/normalizedbaseballcard.csv");
    }
}